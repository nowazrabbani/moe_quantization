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
