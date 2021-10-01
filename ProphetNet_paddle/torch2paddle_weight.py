import paddle
import torch
import numpy as np

# torch_model_path = "cnndm/finetune_cnndm_checkpoints/checkpoint9.pt"
torch_model_path = "pretrained_checkpoints/prophetnet_en.pt"
# torch_model_path = "gigaword/finetune_gigaword_checkpoints/checkpoint7.pt"
torch_state_dict = torch.load(torch_model_path)["model"]

# paddle_model_path = "cnndm/finetune_cnndm_checkpoints/prophetnet-large-uncased-cnndm.pdparams"
paddle_model_path = "pretrained_checkpoints/prophetnet-large-uncased.pdparams.pdparams"
# paddle_model_path = "gigaword/finetune_gigaword_checkpoints/prophetnet-large-uncased-squad-qg.pdparams"
paddle_state_dict = {}

for torch_key in torch_state_dict:
    if "version" in torch_key:
        pass
    else:
        if ('fc' in torch_key) or \
                ('proj' in torch_key and 'ngram_self_attn.in_proj_weight' not in torch_key) or \
                ('relative_linear' in torch_key):
            paddle_state_dict[torch_key] = paddle.to_tensor(torch_state_dict[torch_key].numpy().transpose())
        else:
            paddle_state_dict[torch_key] = paddle.to_tensor(torch_state_dict[torch_key].numpy())

print(len(paddle_state_dict))
paddle.save(paddle_state_dict, paddle_model_path)