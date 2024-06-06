import torch
from enformer_pytorch import seq_indices_to_one_hot
from enformer_pytorch import Enformer
import pandas as pd
import torch.nn as nn
from einops.layers.torch import Rearrange


model = Enformer.from_hparams(
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 5313, mouse = 1643),
    target_length = 896,
)
model.eval()




# Define the model class
# class Conv1DModel(nn.Module):
#     def __init__(self):
#         super(Conv1DModel, self).__init__()
#         # self.rearrange = Rearrange('b n d -> b d n')
#         # Define the convolutional layer
#         self.conv1 = nn.Conv2d(
#             4,#in channel
#             768,#out channel originally 768
#             (1,15),#kernel size
#             padding=(0,7))#padding

#         self.conv2 = nn.Conv2d(
#             4,
#             768,
#             (1,15),
#             padding=(0,7)
#         )


#     def forward(self, x):
#         # half_size = x.shape[3]//2#cannot convert to in16 or will overflow
#         # x1 = self.conv1(x[:,:,:,:half_size])
#         # x2 = self.conv2(x[:,:,:,half_size:])
#         # print(x[:,:,:,:half_size].shape)
#         # print(x[:,:,:,half_size:].shape)
#         # print(x1.shape)
#         # print(x2.shape)

#         return x[:,:,:,98304:]

# model = Conv1DModel()#try to get con1d to work, and fix number -> try rest of trunk
# seq = torch.randint(0, 5, (1, 196_608)) # for ACGTN, in that order (-1 for padding)
seq = torch.ones(1,196_608, dtype=torch.long)#98304 also worked
one_hot=seq_indices_to_one_hot(seq).float()
print(one_hot.shape)
one_hot = one_hot.reshape(1,196608,4)
# one_hot_t = one_hot.transpose(1,3)# 1,4,1,196608
# one_hot = torch.rand(1,196_608,768)
output = model(one_hot)
print(output.shape)

# humanoutput = output['human'] # (1, 896, 5313)
# mouseoutput = output['mouse'] # (1, 896, 1643)
# print(humanoutput.shape)
# print(mouseoutput.shape)


#humanoutput_np = humanoutput.detach().numpy().reshape(-1, 5313)
#mouseoutput_np = mouseoutput.detach().numpy().reshape(-1, 1643)

#Save the outputs to CSV files
#human_df = pd.DataFrame(humanoutput_np)
#mouse_df = pd.DataFrame(mouseoutput_np)

#human_df.to_csv('human_output.csv', index=False)
#mouse_df.to_csv('mouse_output.csv', index=False)

onnx_file_path = "test_enformer.onnx"
torch.onnx.export(model,one_hot, onnx_file_path,input_names=['input'], output_names=['output'])

