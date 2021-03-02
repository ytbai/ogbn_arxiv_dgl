import torch
from torch import nn

class LinearModel(nn.Module):
  def __init__(
    self,
    d_input,
    d_output,
  ):
  
    super().__init__()
    self.d_input = d_input
    self.d_output = d_output
    self.masked = False

    self.linear = nn.Linear(d_input, d_output)

  def forward(self, graph, input):
    return self.linear(input)