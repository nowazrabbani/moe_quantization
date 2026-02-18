import torch
import torch.nn as nn
import argparse
from datautils import get_loaders
import time
import logging
from modelutils import load_mixtral_model, find_layers
import bitallocutils
from gptq import GPTQ
from quant.QLinear import *
from transformers import AutoTokenizer

expert_modules = [
    "block_sparse_moe.experts.0.w1",
    "block_sparse_moe.experts.1.w1",
    "block_sparse_moe.experts.2.w1",
    "block_sparse_moe.experts.3.w1",
    "block_sparse_moe.experts.4.w1",
    "block_sparse_moe.experts.5.w1",
    "block_sparse_moe.experts.6.w1",
    "block_sparse_moe.experts.7.w1",
    "block_sparse_moe.experts.0.w3",
    "block_sparse_moe.experts.1.w3",
    "block_sparse_moe.experts.2.w3",
    "block_sparse_moe.experts.3.w3",
    "block_sparse_moe.experts.4.w3",
    "block_sparse_moe.experts.5.w3",
    "block_sparse_moe.experts.6.w3",
    "block_sparse_moe.experts.7.w3",
    "block_sparse_moe.experts.0.w2",
    "block_sparse_moe.experts.1.w2",
    "block_sparse_moe.experts.2.w2",
    "block_sparse_moe.experts.3.w2",
    "block_sparse_moe.experts.4.w2",
    "block_sparse_moe.experts.5.w2",
    "block_sparse_moe.experts.6.w2",
    "block_sparse_moe.experts.7.w2",
]

logger = logging.getLogger(__name__)

def load_model_and_compute_experts_order(args):

    model, num_layers, num_experts = load_mixtral_model(model_name=args.model,
                                                        attn_implementation=args.attn_implementation
                                                       )

    routers_order, _ = bitallocutils.compute_router_order(
        model,
        num_layers
    )

    if args.order_type == "router":
        return routers_order, model, num_layers, num_experts

    variance_scores = bitallocutils.compute_variance_scores(
        model,
        num_layers,
        num_experts
    )

    if args.order_type == "variance":
        return bitallocutils.compute_variance_order(variance_scores), model, num_layers, num_experts

    if args.order_type == "combined":
        return bitallocutils.compute_combined_order(routers_order,variance_scores,args.zeta), model, num_layers, num_experts

    raise ValueError(
        "order_type must be: router | variance | combined"
    )

@torch.no_grad()
def mixtral_sequential(model, dataloader, dev, args, bit_config=None):
    print('Starting ...')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    print('Ready.')
    quantizers = {}

    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+--------------------------------+------------+------------+------------+---------+')
        print('|              name              |weight_error| fp_inp_SNR | q_inp_SNR  |  time   |')
        print('+================================+============+============+============+=========+')

        layer = layers[i].to(dev)
        full = find_layers(layer)

        sequential = [list(full.keys())]

        if args.mixed_type=="mixed":
            if bit_config is not None:
                low_bit_experts = []
                mid_bit_experts = []
                high_bit_experts = []
                for expert_index in bit_config[i].keys():
                    if bit_config[i][expert_index] == args.low_bit_level:
                        low_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
                    elif bit_config[i][expert_index] == args.high_bit_level:
                        high_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
                    elif args.mid_bit_level is not None:
                        if bit_config[i][expert_index] == args.mid_bit_level:
                            mid_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
                        else:
                            raise ValueError(f"The bit asignment of expert {expert_index} in layer {i} is not in the specified bit levels provided")
                    else:
                        raise ValueError(f"The bit asignment of expert {expert_index} in layer {i} is not in the specified bit levels provided")
            else:
                raise ValueError(f"A bit allocation config is required for mixed-precision quantization")

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                
                gptq[name] = GPTQ(subset[name], logger, name, args.low_bit_level)

                if args.mixed_type == "uniform":
                    if name not in expert_modules:
                        gptq[name].quantizer.configure(args.attn_bit_level, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                        gptq[name].wbits = args.attn_bit_level
                    else:
                        if args.avg_bits.is_integer():
                            gptq[name].quantizer.configure(int(args.avg_bits), perchannel=True, sym=args.sym, mse=False, pack=args.pack) 
                            gptq[name].wbits = int(args.avg_bits)
                        else:
                            raise ValueError("For uniform quantization, avg_bits must be an integer")
                else:
                    if name not in expert_modules:
                        gptq[name].quantizer.configure(args.attn_bit_level, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                        gptq[name].wbits = args.attn_bit_level
                    else:
                        if name[:-3] in high_bit_experts:
                            gptq[name].quantizer.configure(args.high_bit_level, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.high_bit_level
                        elif name[:-3] in low_bit_experts:
                            gptq[name].quantizer.configure(args.low_bit_level, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.low_bit_level
                        else:
                            gptq[name].quantizer.configure(args.mid_bit_level, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.mid_bit_level
            # print(layer)
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['model.layers.%d.%s' % (i, name)] = None
                if args.pack:
                    # real quant for compact memory
                    quant_config = BaseQuantizeConfig(nbits=gptq[name].wbits, group_size=args.groupsize)
                    name_parts = name.split('.')
                    if len(name_parts) == 2: # atten layer
                        _module = getattr(layer, name_parts[-2])
                        linear_layer = getattr(_module, name_parts[-1])
                    else: 
                        experts = getattr(layer.block_sparse_moe, "experts")
                        _module = experts[int(name_parts[-2])]
                        linear_layer = getattr(_module, name_parts[-1])
                    quant_layer = QLinear(quant_config=quant_config, device=linear_layer.weight.device)
                    quant_layer.replace_quantized_weight(linear_layer.weight, scale, zero)
                    setattr(_module, name_parts[-1], quant_layer)
                    print(getattr(_module, name_parts[-1]).W_q.dtype)
                gptq[name].free()
            
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print('+--------------------------------+------------+------------+------------+---------+')
        print('\n')

    model.config.use_cache = use_cache

    return quantizers

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Mixtral8x7B or Mixtral8x22B",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )

    parser.add_argument(
        "--mixed_type",
        type=str,
        choices=["uniform", "mixed"],
        help='Whether to use mixed-precision',
    )

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

    parser.add_argument(
        "--avg_bits",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--high_bit_level",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--mid_bit_level",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--low_bit_level",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--attn_bit_level",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help=" Quantization Group size",
    )

    parser.add_argument(
        '--sym', 
        action='store_true', 
        help='Whether to perform symmetric quantization.'
    )

    parser.add_argument(
        '--act-order', 
        action='store_true', 
        help='Whether to apply the activation order GPTQ heuristic'
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )

    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )

    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )

    parser.add_argument(
        "--eval_ppl", action="store_true", help="Evaluate perplexity."
    )

    parser.add_argument(
        "--pack", action="store_true", help="Whether to save the packed model."
    )

    parser.add_argument(
        "--save",
        action="store_true",
    )

    parser.add_argument(
        "--saving_path", type=str, help="the saving path of quantized model"
    )

    args = parser.parse_args()

    experts_order, model, num_layers, num_experts = load_model_and_compute_experts_order(args)
    
    bit_alloc = bitallocutils.create_bit_distribution(
        experts_order,
        b_avg=args.avg_bits,
        bh=args.high_bit_level,
        bm=args.mid_bit_level,
        bl=args.low_bit_level
    )
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model_map = {
        "Mixtral8x7B": "mistralai/Mixtral-8x7B-v0.1",
        "Mixtral8x22B": "mistralai/Mixtral-8x22B-v0.1",
    }

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=model_map[args.model],
        seqlen=model.seqlen,
    )

    device = "cuda:0"
    tick = time.time()
    mixtral_sequential(model, dataloader, device, args, bit_config=bit_alloc)
    print("quantization time:", time.time() - tick, "s")
    print(model)

    if args.eval_ppl:
        for dataset in ["wikitext2"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, seqlen=2048, model=model_map[args.model]
            )
            print(dataset)
            from eval_ppl_utils import llama_eval
            t1 = time.time()
            llama_eval(model, testloader, device, dataset)
            print("Time: ", time.time() - t1)

    if args.save:
        saving_path = args.saving_path
        tokenizer = AutoTokenizer.from_pretrained(model_map[args.model])
        tokenizer.save_pretrained(saving_path)
        from utils.pack import save_quantized
        save_quantized(model, saving_path)
    

if __name__ == "__main__":
    main()