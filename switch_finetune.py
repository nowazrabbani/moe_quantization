import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DistributedType, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
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

model_checkpoint = "google/switch-base-64"

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
        
        target = self.tokenizer([targets], max_length= self.target_len, padding='max_length', truncation=True, return_tensors='pt')
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long),
        }

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-64")

train_dataset=encoded_dataset["train"]
sequence_length = {"inputs": 512, "targets": 512}
training_set = CustomDataset(train_dataset, tokenizer, sequence_length["inputs"], sequence_length["targets"])

def create_dataloaders(train_batch_size=16):
    train_dataloader = DataLoader(
        training_set, shuffle=True, batch_size=train_batch_size
    )
    return train_dataloader

from transformers import SwitchTransformersForConditionalGeneration

model = SwitchTransformersForConditionalGeneration.from_pretrained(model_checkpoint)
model.config.decoder_start_token_id=0

hyperparameters = {
    "learning_rate": 0.00018,
    "num_epochs": 34,
    "train_batch_size": 8, # Actual batch size will this x 8
}

def training_function(model):
    # Initialize accelerator
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy='transformer_based_wrap',
        transformer_cls_names_to_wrap='SwitchTransformersBlock',
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
        use_orig_params=True,
    )
    accelerator = Accelerator(log_with="tensorboard", project_dir="saved_train_directory/cnn_dm/",fsdp_plugin=fsdp_plugin)
    accelerator.init_trackers("train_directory", config=hyperparameters)

    # To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    train_dataloader = create_dataloaders(train_batch_size=hyperparameters["train_batch_size"])
    # The seed need to be set before we instantiate the model, as it will determine the random head.
    #set_seed(hyperparameters["seed"])

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    num_epochs = hyperparameters["num_epochs"]
    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    #lr_scheduler = get_linear_schedule_with_warmup(
        #optimizer=optimizer,
        #num_warmup_steps=100,
        #num_training_steps=len(train_dataloader) * num_epochs,
    #)

    # Instantiate a progress bar to keep track of training. Note that we only enable it on the main
    # process to avoid having 8 progress bars.
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)), disable=not accelerator.is_main_process)
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            labels = torch.as_tensor(batch['target_ids'], dtype = torch.long)
            ids = torch.as_tensor(batch['source_ids'], dtype = torch.long)
            mask = torch.as_tensor(batch['source_mask'], dtype = torch.long)
            decoder_mask = torch.as_tensor(batch['target_mask'], dtype = torch.long)
    
            outputs = model(input_ids = ids, attention_mask = mask, decoder_attention_mask = decoder_mask, labels=labels, 
                            output_router_logits=True, return_dict=True)
            
            loss = outputs[0]

            accelerator.backward(loss)
            
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
        accelerator.log({"training_loss": loss},
                        step=epoch)
        
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        "switch_finetuned_ckpt",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    accelerator.end_training()

def main():
    training_function(model)

if __name__=="__main__":
    main()