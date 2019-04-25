import torch.nn as nn
import torch



weight = torch.tensor([1, 2, 3, 4, 5, 6], dtype = torch.float)
inputs = torch.tensor([1, 1], dtype = torch.float)

model = nn.Linear(2, 3, bias = False)

model.weight.data = weight.reshape(model.weight.shape)

out = model(inputs)

print(out)

