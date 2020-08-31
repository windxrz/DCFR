import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, shapes, acti):
        super(MLP, self).__init__()
        self.acti = acti
        self.fc = nn.ModuleList()
        for i in range(0, len(shapes) - 1):
            self.fc.append(nn.Linear(shapes[i], shapes[i + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i == len(self.fc) - 1:
                break
            if self.acti == "relu":
                x = F.relu(x)
            elif self.acti == "sigmoid":
                x = F.sigmoid(x)
            elif self.acti == "softplus":
                x = F.softplus(x)
            elif self.acti == "leakyrelu":
                x = F.leaky_relu(x)
        return x

    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def activate(self):
        for para in self.parameters():
            para.requires_grad = True
