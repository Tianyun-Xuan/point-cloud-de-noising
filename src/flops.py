import torch
import numpy as np
from mistnet import MistNet

print(torch.__version__)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(device)


model = MistNet(4, 4)
model.load_state_dict(torch.load('/home/mist/model.pth'))

model.eval().to(device)

Total_params = 0
Trainable_params = 0
NonTrainable_params = 0

for param in model.parameters():
    mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    Total_params += mulValue  # 总参数量
    if param.requires_grad:
        Trainable_params += mulValue  # 可训练参数量
    else:
        NonTrainable_params += mulValue  # 非可训练参数量

print(f'Total params: {Total_params / 1e6}M')
print(f'Trainable params: {Trainable_params / 1e6}M')
print(f'Non-trainable params: {NonTrainable_params / 1e6}M')

# calculate the model flops

dummy_input = torch.randn(1, 4, 128, 1200).to(device)
from thop import profile
flops, params = profile(model, inputs=(dummy_input, ))
print(f"flops is {flops / 1e9}G")