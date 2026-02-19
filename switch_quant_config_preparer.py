import pickle
import torch
import copy
from transformers import SwitchTransformersForConditionalGeneration, HqqConfig
from hqq.core.quantize import *

def compute_expert_order(
    model_checkpoint: str,
    order_type: str = "router",
    zeta: float = 3.0,
):
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        model_checkpoint,
        device_map='cpu',
    )
    
    moe_layers = [1, 3, 5, 7, 9, 11]
    
    encoder_router_order = []
    decoder_router_order = []

    for i in moe_layers:
        en_weight = model.state_dict()[
            f"encoder.block.{i}.layer.1.mlp.router.classifier.weight"
        ]
        de_weight = model.state_dict()[
            f"decoder.block.{i}.layer.2.mlp.router.classifier.weight"
        ]

        _, en_indices = torch.sort(
            torch.linalg.vector_norm(en_weight, ord=2, dim=1)
        )
        _, de_indices = torch.sort(
            torch.linalg.vector_norm(de_weight, ord=2, dim=1)
        )

        encoder_router_order.append(en_indices.tolist())
        decoder_router_order.append(de_indices.tolist())

    if order_type == "router":
        del model
        return encoder_router_order, decoder_router_order

    en_var = []
    de_var = []

    for i in moe_layers:
        en_layer_var = []
        de_layer_var = []

        for j in range(64):
            en_w = model.state_dict()[
                f"encoder.block.{i}.layer.1.mlp.experts.expert_{j}.wi.weight"
            ]
            de_w = model.state_dict()[
                f"decoder.block.{i}.layer.2.mlp.experts.expert_{j}.wi.weight"
            ]

            en_layer_var.append(torch.max(torch.var(en_w, dim=1)))
            de_layer_var.append(torch.max(torch.var(de_w, dim=1)))

        en_var.append(en_layer_var)
        de_var.append(de_layer_var)

    if order_type == "variance":
        encoder_var_order = [
            torch.argsort(torch.tensor(v)).tolist() for v in en_var
        ]
        decoder_var_order = [
            torch.argsort(torch.tensor(v)).tolist() for v in de_var
        ]
        del model
        return encoder_var_order, decoder_var_order

    if order_type == "combined":

        encoder_order_var = []
        decoder_order_var = []

        for i in range(6):

            c_l = copy.deepcopy(encoder_router_order[i])

            for j in reversed(encoder_router_order[i]):
                temp = [
                    k
                    for k, val in enumerate([en_var[i][l] for l in c_l])
                    if (zeta * val) < en_var[i][j]
                ]
                if temp and c_l.index(j) > temp[0]:
                    y = c_l.pop(c_l.index(j))
                    c_l.insert(temp[0], y)

            encoder_order_var.append(c_l)

            c_d = copy.deepcopy(decoder_router_order[i])

            for j in reversed(decoder_router_order[i]):
                temp_d = [
                    k_d
                    for k_d, val_d in enumerate([de_var[i][d] for d in c_d])
                    if (zeta * val_d) < de_var[i][j]
                ]
                if temp_d and c_d.index(j) > temp_d[0]:
                    y_d = c_d.pop(c_d.index(j))
                    c_d.insert(temp_d[0], y_d)

            decoder_order_var.append(c_d)
        del model
        return encoder_order_var, decoder_order_var

    raise ValueError(
        "order_type must be one of: 'router', 'variance', 'combined'"
    )

group_quant_configs_wi = [{'nbits':1,'group_size':None,'axis':1},{'nbits':2,'group_size':None,'axis':1}, {'nbits':3,'group_size':768,'axis':1}, 
                       {'nbits':4,'group_size':None,'axis':1}, {'nbits':8,'group_size':None,'axis':1}]
group_quant_configs_wo = [{'nbits':1,'group_size':None,'axis':1},{'nbits':2,'group_size':None,'axis':1}, {'nbits':3,'group_size':3072,'axis':1}, 
                       {'nbits':4,'group_size':None,'axis':1}, {'nbits':8,'group_size':None,'axis':1}]

no_of_experts = 64

def create_quant_configs(model_checkpoint, order_type, zeta, encoder_expert_quant_group_sizes, decoder_expert_quant_group_sizes):
    encoder_order, decoder_order = compute_expert_order(model_checkpoint, order_type=order_type, zeta=zeta)
    module_configs= pickle.load(open('switch_pre_module_configs.pkl','rb'))
    for i in [1,3,5,7,9,11]:
        for j in range(no_of_experts):
            if j<=encoder_expert_quant_group_sizes[(i-1)//2][0]-1:
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[0]})
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[0]})
            elif j<=encoder_expert_quant_group_sizes[(i-1)//2][0]+encoder_expert_quant_group_sizes[(i-1)//2][1]-1:
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[1]})
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[1]})
            elif j<=(encoder_expert_quant_group_sizes[(i-1)//2][0]+encoder_expert_quant_group_sizes[(i-1)//2][1]+
                    encoder_expert_quant_group_sizes[(i-1)//2][2]-1):
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[2]})
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[2]})
            elif j<=(encoder_expert_quant_group_sizes[(i-1)//2][0]+encoder_expert_quant_group_sizes[(i-1)//2][1]+
                    encoder_expert_quant_group_sizes[(i-1)//2][2]+encoder_expert_quant_group_sizes[(i-1)//2][3]-1):
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[3]})
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[3]})
            else:
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[4]})
                module_configs.update({'encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+
                                       str(encoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[4]})
            
    for i in [1,3,5,7,9,11]:
        for j in range(no_of_experts):
            if j<=decoder_expert_quant_group_sizes[(i-1)//2][0]-1:
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[0]})
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[0]})
            elif j<=decoder_expert_quant_group_sizes[(i-1)//2][0]+decoder_expert_quant_group_sizes[(i-1)//2][1]-1:
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[1]})
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[1]})
            elif j<=(decoder_expert_quant_group_sizes[(i-1)//2][0]+decoder_expert_quant_group_sizes[(i-1)//2][1]+
                decoder_expert_quant_group_sizes[(i-1)//2][2]-1):
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[2]})
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[2]})
            elif j<=(decoder_expert_quant_group_sizes[(i-1)//2][0]+decoder_expert_quant_group_sizes[(i-1)//2][1]+
                decoder_expert_quant_group_sizes[(i-1)//2][2]+decoder_expert_quant_group_sizes[(i-1)//2][3]-1):
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[3]})
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[3]})
            else:
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wi':group_quant_configs_wi[4]})
                module_configs.update({'decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+
                                       str(decoder_order[(i-1)//2][63-j])+'.wo':group_quant_configs_wo[4]})
    no_of_blocks = 6
    total_bits = 0.0
    precisions = [1,2,3,4,8]
    for i in range(no_of_blocks):
        for j in range(5):
            total_bits = total_bits+(encoder_expert_quant_group_sizes[i][j]*precisions[j])+(decoder_expert_quant_group_sizes[i][j]*precisions[j])
    avg_bits_per_expert = total_bits/(no_of_experts*no_of_blocks*2)
    return avg_bits_per_expert, module_configs