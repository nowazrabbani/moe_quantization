# Code for the paper titled "Efficient Quantization of Mixture-of-Experts with Theoretical Generalization Guarantees" [ICLR'2026]
This repository implements post-training weight quantization of large MoE models, including Mixtral and Switch Transformer.

## Installation

Follow the steps below to set up the environment and install dependencies:

```bash
conda create --name moe_quant
conda activate moe_quant

git clone https://github.com/nowazrabbani/moe_quantization.git
cd moe_quantization
pip install -r requirements.txt

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

mv huggingface.py lm-evaluation-harness/lm_eval/models
mkdir lm-evaluation-harness/lm_eval/quantized
mv inference.py lm-evaluation-harness/lm_eval/quantized
scp -r quant lm-evaluation-harness/lm_eval/quantized
scp -r utils lm-evaluation-harness/lm_eval/quantized
```

## Quantizing and Saving Mixtral Model

Use the following script to perform post-training weight quantization of the Mixtral MoE model and save the quantized checkpoint.

### Example Command

```bash
python quantize_mixtral.py \
  --model Mixtral8x7B \
  --mixed_type mixed \
  --order_type combined \
  --zeta 3.0 \
  --avg_bits 2.0 \
  --high_bit_level 3 \
  --mid_bit_level 2 \
  --low_bit_level 1 \
  --attn_bit_level 4 \
  --groupsize 128 \
  --dataset wikitext2 \
  --eval_ppl \
  --pack \
  --save \
  --saving_path quantized_ckpts/2p0
```

### Arguments

* `--model` : Mixtral model variant to quantize (e.g., `Mixtral8x7B`).
* `--mixed_type` : Quantization type configuration (`mixed` for mixed-precision).
* `--order_type` : Router norm, MaxVar, or combination based expert ranking (e.g., `combined` for router norm based ordering and MaxVar based reordering).
* `--zeta` : Hyperparameter controlling the expert reordering frequency based on MaxVar.
* `--avg_bits` : Target average bit-width/expert in an MoE layer.
* `--high_bit_level` : Bit precision for high-bit experts.
* `--mid_bit_level` : Bit precision for mid-bit experts.
* `--low_bit_level` : Bit precision for low-bit experts.
* `--attn_bit_level` : Bit precision for attention layers.
* `--groupsize` : Group size for GPTQ.
* `--dataset` : Calibration dataset (e.g., `wikitext2`).
* `--eval_ppl` : Whether to evaluate perplexity after quantization.
* `--pack` : Whether to pack quantized weights.
* `--save` : Whether to save quantized checkpoint.
* `--saving_path` : Path to store the quantized model.

### Output

The quantized Mixtral checkpoint is saved to the directory specified by `--saving_path`.
This checkpoint can be used for downstream evaluation or inference with the LM Evaluation Harness.

## Inference on Quantized Mixtral Model

After quantizing and saving the Mixtral checkpoint, you can run downstream evaluation using the LM Evaluation Harness.

### Example Command

```bash
lm_eval --model hf \
  --tasks piqa,boolq,arc_challenge,arc_easy,hellaswag,winogrande,mmlu,mathqa \
  --model_args pretrained='quantized_ckpts/2p0',parallelize=True,quantized=True,dtype='float16' \
  --batch_size 32
```

### Arguments

* `--model hf` : Uses the Hugging Face model interface.
* `--tasks` : Comma-separated list of evaluation tasks.
* `--model_args` :

  * `pretrained` : Path to the quantized checkpoint directory.
  * `parallelize=True` : Enables multi-GPU inference (if available).
  * `quantized=True` : Loads the model in quantized mode.
  * `dtype='float16'` : Sets computation precision.
* `--batch_size` : Batch size for evaluation.

### Output

The command reports task-wise evaluation metrics (e.g., accuracy) and overall aggregated performance of the quantized Mixtral model.


