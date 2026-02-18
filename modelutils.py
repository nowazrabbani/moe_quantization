import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoConfig

def load_mixtral_model(model_name: str,
                      attn_implementation: str,
                      ):

    model_map = {
        "Mixtral8x7B": "mistralai/Mixtral-8x7B-v0.1",
        "Mixtral8x22B": "mistralai/Mixtral-8x22B-v0.1",
    }
    
    if model_name not in model_map:
        raise ValueError("model_name must be 'Mixtral8x7B' or 'Mixtral8x22B'")

    config = AutoConfig.from_pretrained(
        model_map[model_name], attn_implementation=attn_implementation
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_map[model_name],
        config=config,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    model.seqlen = 2048

    # Architecture constants
    if model_name == "Mixtral8x7B":
        num_layers = 32
        num_experts = 8
    elif model_name == "Mixtral8x22B":
        num_layers = 56
        num_experts = 8

    return model, num_layers, num_experts

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


