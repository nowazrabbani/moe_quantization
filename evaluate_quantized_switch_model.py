import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from tqdm.auto import tqdm
import functools

import datasets
import transformers
import evaluate
import numpy as np
from absl import logging

model_checkpoint = "switch_finetuned_ckpt"

raw_datasets = load_dataset("cnn_dailymail", "3.0.0")

def preprend_cnndm(example):
      return {"article":"summarize: "+ example['article']}
encoded_dataset = raw_datasets.map(preprend_cnndm, batched=False)
encoded_dataset = encoded_dataset.rename_column("article", "context")
encoded_dataset = encoded_dataset.rename_column("highlights", "targets")
encoded_dataset = encoded_dataset.remove_columns("id")

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.context = self.data["context"]
        self.targets = self.data["targets"]
        

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        context = self.context[index]
        targets = self.targets[index]
        
        source = self.tokenizer([context], max_length= self.source_len, padding='max_length', truncation=True, return_tensors='pt')
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'targets': targets
        }

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-64")

val_dataset=encoded_dataset["validation"]
sequence_length = {"inputs": 512, "targets": 512}
val_set = CustomDataset(val_dataset, tokenizer, sequence_length["inputs"], sequence_length["targets"])

from rouge_score import rouge_scorer
from rouge_score import scoring
import collections

def _prepare_summary_rouge(summary):
  # Make sure the summary is not bytes-type
  # Add newlines between sentences so that rougeLsum is computed correctly.
  summary = summary.replace(" . ", " .\n")
  return summary

def rouge(
    targets,
    predictions,
    score_keys=("rouge1", "rouge2", "rougeLsum"),
    verbose=False,
    **kwargs,
):
  """Computes rouge score nondeterministically using the bootstrap.

  Args:
    targets: list of strings.
    predictions: list of strings.
    score_keys: list of strings with the keys to compute.
    verbose: whether to enable additional logging.
    **kwargs: additional keyword arguments for RougeScorer.

  Returns:
    dict with score_key: rouge score across all targets and predictions
  """

  scorer = rouge_scorer.RougeScorer(rouge_types=score_keys, **kwargs)
  aggregator = scoring.BootstrapAggregator()

  for prediction, target in zip(predictions, targets):
    target = _prepare_summary_rouge(target)
    prediction = _prepare_summary_rouge(prediction)
    aggregator.add_scores(scorer.score(target=target, prediction=prediction))
  result = aggregator.aggregate()
  if verbose:
    for key in score_keys:
      logging.info(
          "%s = %.2f, 95%% confidence [%.2f, %.2f]",
          key,
          result[key].mid.fmeasure*100,
          result[key].low.fmeasure*100,
          result[key].high.fmeasure*100,
      )
  return {key: result[key].mid.fmeasure*100 for key in score_keys}


def rouge_mean(
    targets,
    predictions,
    score_keys=("rouge1", "rouge2", "rougeLsum"),
    **kwargs,
):
  """Computes rouge score deterministically (no bootstrap).

  Args:
    targets: list of strings
    predictions: list of strings
    score_keys: list of strings with the keys to compute
    **kwargs: additional keyword arguments for RougeScorer.

  Returns:
    dict with score_key: rouge score across all targets and predictions
  """

  scorer = rouge_scorer.RougeScorer(rouge_types=score_keys, **kwargs)
  count = 0
  sum_scores = collections.defaultdict(float)
  for prediction, target in zip(predictions, targets):
    target = _prepare_summary_rouge(target)
    prediction = _prepare_summary_rouge(prediction)
    scores = scorer.score(target=target, prediction=prediction)
    count += 1
    for k, v in scores.items():
      sum_scores[k] += v.fmeasure
  if count == 0:
    raise ValueError("Predictions and targets must both have nonzero length")
  result = {k: v / count for k, v in sum_scores.items()}
  return {key: result[key] * 100 for key in score_keys}


def evaluate_cnndm_nd(targets, predictions):
    return rouge(targets, predictions)

def evaluate_cnndm_d(targets, predictions):
    return rouge_mean(targets, predictions)

from transformers import SwitchTransformersForConditionalGeneration, HqqConfig
from accelerate import PartialState
state = PartialState()
from accelerate.utils import gather_object

eval_dataloader = DataLoader(val_set, shuffle=False, batch_size=128)

def evaluation_function(module_configs):
    quant_config = HqqConfig(dynamic_config=module_configs)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(model_checkpoint,device_map="cuda",quantization_config=quant_config,
                                                                   torch_dtype=torch.float32)
    model.eval()
    all_predictions = []
    all_labels = []
    for step, batch in enumerate(eval_dataloader):
        with state.split_between_processes(batch) as inputs:
            inputids = torch.as_tensor(inputs['source_ids'], dtype = torch.long, device=state.device)
            mask = torch.as_tensor(inputs['source_mask'], dtype = torch.long, device=state.device)
            with torch.no_grad():
                outputs = model.generate(input_ids = inputids,attention_mask=mask,max_new_tokens=512,decoder_start_token_id=0)
            string_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        # We gather predictions and labels from the 8 TPUs to have them all.
        predictions_gather = gather_object(string_predictions)
        label_gather = gather_object(batch['targets'])
        all_predictions.append(predictions_gather)
        all_labels.append(label_gather)
    
    # Concatenate all predictions and labels.
    # The last thing we need to do is to truncate the predictions and labels we concatenated
    # together as the prepared evaluation dataloader has a little bit more elements to make
    # batches of the same size on each process.
    all_predictions = np.concatenate(all_predictions)[:len(val_set)]
    all_labels = np.concatenate(all_labels)[:len(val_set)]
    eval_metric_nd = evaluate_cnndm_nd(all_labels, all_predictions)
    eval_metric_d = evaluate_cnndm_d(all_labels, all_predictions)
    return eval_metric_nd, eval_metric_d

import switch_quant_config_preparer
import math

def build_expert_group_sizes(
    high_bit_level,
    low_bit_level,
    n_high,
    n_low,
):

    bit_to_index = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        8: 4,
    }

    if high_bit_level not in bit_to_index:
        raise ValueError(f"Unsupported high bit level: {high_bit_level}")

    if low_bit_level not in bit_to_index:
        raise ValueError(f"Unsupported low bit level: {low_bit_level}")

    # Base template per layer
    layer_template = [0, 0, 0, 0, 0]

    layer_template[bit_to_index[low_bit_level]] = n_low
    layer_template[bit_to_index[high_bit_level]] = n_high

    # Replicate for all MoE layers
    return [layer_template.copy() for _ in range(6)]

def quantize(args):
    encoder_expert_quant_group_sizes = build_expert_group_sizes(args.high_bit_level, 
                                                                args.low_bit_level, 
                                                                args.num_high_bit_experts, 
                                                                args.num_low_bit_experts)
    decoder_expert_quant_group_sizes = build_expert_group_sizes(args.high_bit_level, 
                                                                args.low_bit_level, 
                                                                args.num_high_bit_experts, 
                                                                args.num_low_bit_experts)
    avg_bits , module_configs = switch_quant_config_preparer.create_quant_configs(model_checkpoint,
                                                                                  args.order_type,
                                                                                  args.zeta,
                                                                                  encoder_expert_quant_group_sizes, 
                                                                                  decoder_expert_quant_group_sizes)
    eval_metric_nd, eval_metric_d = evaluation_function(module_configs)
    state.print(f"(Nondeterministic) Rouge-1: ", eval_metric_nd['rouge1'])
    state.print(f"(Nondeterministic) Rouge-2: ", eval_metric_nd['rouge2'])
    state.print(f"(Nondeterministic) Rouge-L: ", eval_metric_nd['rougeLsum'])
    state.print(f"(Deterministic) Rouge-1: ", eval_metric_d['rouge1'])
    state.print(f"(Deterministic) Rouge-2: ", eval_metric_d['rouge2'])
    state.print(f"(Deterministic) Rouge-L: ", eval_metric_d['rougeLsum'])
    state.print(f'Avg bits:',avg_bits)
    
def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()

    parser.add_argument("--high_bit_level", type=int, required=True)
    parser.add_argument("--low_bit_level", type=int, required=True)

    parser.add_argument("--num_low_bit_experts", type=int, default=8)
    parser.add_argument("--num_high_bit_experts", type=int, default=56)

    parser.add_argument(
        "--order_type",
        type=str,
        required=True,
        help="router | variance | combined",
    )

    parser.add_argument(
        "--zeta",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()
    
    quantize(args)

if __name__=="__main__":
    main()
