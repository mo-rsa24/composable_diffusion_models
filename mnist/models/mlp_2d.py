# models/mlp_2d.py
import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, num_hid=128, num_out=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(1 + num_out, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_hid),
            nn.SiLU(),
            nn.Linear(num_hid, num_out)
        )
    def forward(self, t, x):
        t_in = t.view(-1, 1)
        h = torch.cat([t_in, x], dim=1)
        return self.main(h)