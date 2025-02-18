import torch
from torch import nn

class RHN(nn.Module):
    def __init__(self, inputnodes, hiddennodes):
        super(RHN, self).__init__()
        self.xnodes = inputnodes
        self.hnodes = hiddennodes
        self.w_xh = nn.Parameter(torch.randn(self.xnodes, self.hnodes) * 0.1, requires_grad=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        if x.shape[1] != self.w_xh.shape[0]:
            raise ValueError(f"Dimension mismatch: input {x.shape}, weight {self.w_xh.shape}")
        return torch.tanh(x @ self.w_xh) @ self.w_xh.T

    def loss(self, y, y_pred):
        return torch.nn.functional.mse_loss(y, y_pred)
