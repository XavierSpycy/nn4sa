import torch
import torch.nn as nn

class CLSHead(nn.Module):
    def __init__(self, 
                 output_dim: int,
                 input_dim) -> None:
        super(CLSHead, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim)
            )
    
    def forward(self, x) -> torch.Tensor:
        return self.fc(x)