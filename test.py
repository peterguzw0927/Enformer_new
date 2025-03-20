import torch
from fresh.enformer_pytorch_fresh.enformer_pytorch_fresh import seq_indices_to_one_hot
from enformer_pytorch import Enformer,GenomeIntervalDataset
import pandas as pd
import torch.nn as nn
from einops.layers.torch import Rearrange
from fresh.enformer_pytorch_fresh.enformer_pytorch_fresh import from_pretrained
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import polars as pl
from einops import rearrange
import pickle
import matplotlib.pyplot as plt
from enformer_pytorch.metrics import *

# torch.manual_seed(42)
data_path = "/projectnb/aclab/datasets/basenji/expanded/human/196608/train-0.pickle"

enformer = from_pretrained('EleutherAI/enformer-official-rough',
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 5313, mouse = 1643),
    target_length = 896,
    )
enformer = enformer.eval()
pretrained_state_dict=enformer.state_dict()

model = Enformer.from_hparams(
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 5313, mouse = 1643),
    target_length = 896,
)
model = model.eval()
model = model.cuda()
enformer = enformer.cuda()

# load data using pickle load, and each "sequence" is 196k , and then "target" is 896,5313
with open(data_path,'rb') as file:
    data = pickle.load(file)

seq = data.get('sequence',None)
seq = torch.from_numpy(seq)
seq = seq.reshape(1,-1)
one_hot = seq_indices_to_one_hot(seq).float()
target = data.get('target',None).reshape(1,896,5313)
target = torch.from_numpy(target).cuda()


# seq = torch.randint(0, 5, (1, 196_608)) # for ACGTN, in that order (-1 for padding)
# print(seq.shape)
# one_hot=seq_indices_to_one_hot(seq).float()
one_hot = one_hot.cuda()



custom_state_dict=model.state_dict()

torch.save(enformer.state_dict(),"fresh.pth")
state_dict = torch.load("fresh.pth")
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)


with torch.no_grad():
    output = model(one_hot)
    output_real = enformer(one_hot)



# print("output shape: ",output.shape)
humanoutput = output['human']# (1, 896, 5313)
humanoutput_real = output_real['human']


diff = humanoutput_real - humanoutput
diff = torch.sum(diff**2)
print("sum of squared is :", diff)


metric = MeanPearsonCorrCoefPerChannel(5313).cuda()
metric.update(preds=humanoutput,target=target)
r = metric.compute()
print("R value is ",torch.mean(r))
torch.save(humanoutput,"humanoutput.pt")

# onnx_file_path = "test_enformer.onnx"
# torch.onnx.export(model,one_hot, onnx_file_path,input_names=['input'], output_names=['output'],opset_version=12)#opset_version =12 not work





# print("Missing keys:", missing_keys)
# print("Unexpected keys:", unexpected_keys)


# x = model._trunk[0](one_hot)
# x = model._trunk[1](x)
# # print(model._trunk[1])
# x = model._trunk[2](x)
# x = model._trunk[3](x)

# print(orig)
# print("x: \n",x)

# for name, layer in orig.named_modules():
#     # if name in [str(z) for z in range(3)]:
#     layer.register_forward_hook(print_hook(name))

# for name, layer in new.named_modules():
#     # if name in [str(z) for z in range(3)]:
#     layer.register_forward_hook(print_hook(name))

# print("orig: \n",orig(x))
# print("new:\n",new(x))
# print("new:\n",new(x))

# import sys
# sys.exit(0)

# for name, layer in model.trunk.named_modules():
#     if name == "1":
#         layer.register_forward_hook(print_hook(name))
# print("Printing my enformer values:")
# # Register hooks for each layer in the trunk
# transformer_module = model._trunk[4]

# Register hooks for each submodule in the transformer
# i=0
# for submodule_name, submodule in transformer_module.named_modules():
#     submodule.register_forward_hook(print_hook(f"Transformer.{submodule_name}"))
    # i=i+1
    # if i ==2:
    #     break

# for name, layer in model.trunk.named_modules():
#     if name == "1":
#         layer.register_forward_hook(print_hook(name))

# print("Printing fresh enformer values:")
# transformer_module = enformer._trunk[4]

# i=0
# Register hooks for each submodule in the transformer
# for submodule_name, submodule in transformer_module.named_modules():
#     submodule.register_forward_hook(print_hook(f"Transformer.{submodule_name}"))
    # i=i+1
    # if i ==2:
    #     break

# Register hooks for each layer in the trunk
# for name, layer in enformer.trunk.named_modules():
#     if name == "1":
#         layer.register_forward_hook(print_hook(name))

# for name, param in pretrained_state_dict.items():
#     if "PrintModule" in name:
#         print(f"Skipping PrintModule layer: {name}")
#         continue
#     elif name in custom_state_dict and custom_state_dict[name].size() == param.size():
#         custom_state_dict[name].copy_(param)
#         print(f"Copied weights for layer: {name}")
#     else:
#         print(f"Skipping layer: {name} (size mismatch or not found)")

        # Load updated state_dict into your custom model
# model.load_state_dict(torch.load("my_model.pth",weights_only=True))
# torch.save(model.state_dict(), "my_model.pth")
# print("Custom model saved successfully!")



# seq = torch.randint(0, 5, (1, 196_608)) # for ACGTN, in that order (-1 for padding)
# # seq = torch.ones(1,196_608, dtype=torch.long)#98304 also worked
# one_hot=seq_indices_to_one_hot(seq).float()
# # print("input shape: ",one_hot.shape)
# # one_hot = one_hot.reshape(1,196608,4)





# Check for NaN values
# contains_nan = torch.isnan(humanoutput)
# contains_nan_real = torch.isnan(humanoutput_real)
# not_nan = ~contains_nan
# # print("My enformer output:",humanoutput[0:10,0])
# # print("True enformer output:",humanoutput_real[0:10,0])
# print("My enformer Contains NaN:", torch.nonzero(contains_nan, as_tuple=False))
# print("True enforemer Conains NaN",torch.nonzero(contains_nan_real, as_tuple=False))


