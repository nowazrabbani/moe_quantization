finetuned_model_checkpoint = "switch_cnndm"
from transformers import SwitchTransformersForConditionalGeneration
pretrained_model = SwitchTransformersForConditionalGeneration.from_pretrained('google/switch-base-128')
finetuned_model = SwitchTransformersForConditionalGeneration.from_pretrained(finetuned_model_checkpoint)


encoder_order = []
encoder_values = []
for i in [1,3,5,7,9,11]:
    values,indices = torch.sort(torch.linalg.vector_norm(finetuned_model.state_dict()['encoder.block.'+str(i)+'.layer.1.mlp.router.classifier.weight'],
                                                   ord=2,dim=1)-
                         torch.linalg.vector_norm(pretrained_model.state_dict()['encoder.block.'+str(i)+'.layer.1.mlp.router.classifier.weight'],
                                                  ord=2,dim=1))
    encoder_order.append(indices.tolist())
    encoder_values.append(values.tolist())

decoder_order = []
decoder_values = []
for i in [1,3,5,7,9,11]:
    values,indices = torch.sort(torch.linalg.vector_norm(finetuned_model.state_dict()['decoder.block.'+str(i)+'.layer.2.mlp.router.classifier.weight'],
                                                   ord=2,dim=1)-
                         torch.linalg.vector_norm(pretrained_model.state_dict()['decoder.block.'+str(i)+'.layer.2.mlp.router.classifier.weight'],
                                                  ord=2,dim=1))
    decoder_order.append(indices.tolist())
    decoder_values.append(values.tolist())

import torch
en_var = []
for i in [1,3,5,7,9,11]:
    y = []
    for j in range(64):
        y.append(torch.max(torch.var(finetuned_model.state_dict()['encoder.block.'+str(i)+'.layer.1.mlp.experts.expert_'+str(j)+'.wi.weight'],dim=1)))
    en_var.append(y)


de_var = []
for i in [1,3,5,7,9,11]:
    y = []
    for j in range(64):
        y.append(torch.max(torch.var(fiunetuned_model.state_dict()['decoder.block.'+str(i)+'.layer.2.mlp.experts.expert_'+str(j)+'.wi.weight'],dim=1)))
    de_var.append(y)

import copy
encoder_order_var = []
for i in range(6):
    c_l = copy.deepcopy(encoder_order[i])
    for j in reversed(encoder_order[i]):
        temp=[k for k, val in enumerate([en_var[i][l] for l in c_l]) if (3*val)<en_var[i][j]]
        if temp:
            if c_l.index(j)>temp[0]:
                y = c_l.pop(c_l.index(j))
                c_l.insert(temp[0],y)
    encoder_order_var.append(c_l)

decoder_order_var = []
for i in range(6):
    c_d = copy.deepcopy(decoder_order[i])
    for j in reversed(decoder_order[i]):
        temp_d=[k_d for k_d, val_d in enumerate([de_var[i][d] for d in c_d]) if (3*val_d)<de_var[i][j]]
        if temp_d:
            if c_d.index(j)>temp_d[0]:
                y_d = c_d.pop(c_d.index(j))
                c_d.insert(temp_d[0],y_d)
    decoder_order_var.append(c_d)

dict_file_1 = open('experts_encoder_order.pkl','wb')
pickle.dump(encoder_order_var,dict_file_1)
dict_file_1.close()

dict_file_2 = open('experts_decoder_order.pkl','wb')
pickle.dump(decoder_order_var,dict_file_2)
dict_file_2.close()