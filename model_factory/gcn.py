import torch
from torch import nn
import numpy as np
from dgl.nn.pytorch.conv import GraphConv


class GCNLayer(nn.Module):
  def __init__(self, d_model, add_linear):
    super().__init__()
    self.d_model = d_model
    self.add_linear = add_linear
    self.drop_p = 0.5

    if add_linear:
      self.linear = nn.Linear(d_model, d_model)
    
    self.gcn = GraphConv(d_model, d_model)


    self.top = nn.Sequential(
                              nn.BatchNorm1d(d_model),
                              nn.ReLU(),
                              nn.Dropout(self.drop_p),
    )

  
  def forward(self, graph, input):
    output = self.gcn(graph, input)

    output += input

    if self.add_linear:
      output += self.linear(input)

    output = self.top(output)

    return output


class GCNHead(nn.Module):
  def __init__(self, d_model, n_layer, add_linear):
    super().__init__()
    self.d_model = d_model
    self.n_layer = n_layer
    self.add_linear = add_linear

    self.layer_list = nn.ModuleList([
                                 GCNLayer(d_model, add_linear) for _ in range(n_layer)
    ])

  def forward(self, graph, input):
    output = input

    for layer in self.layer_list:
      output = layer(graph, output)
    
    return output


class GCN(nn.Module):
  def __init__(self, d_input, n_class, d_model, n_layer, n_head, masked, add_linear=True):
    super().__init__()
    self.d_input = d_input
    self.d_model = d_model
    self.n_class = n_class
    self.n_layer = n_layer
    self.n_head = n_head
    self.add_linear = add_linear
    self.masked = masked
    self.input_drop_p = 0.1


    self.input_embed = nn.Linear(d_input, d_model)
    self.input_dropout = nn.Dropout(self.input_drop_p)
  
    self.head_list = nn.ModuleList([
                                     GCNHead(d_model, n_layer, add_linear) for _ in range(n_head)
                                     ])
    self.top = nn.Sequential(
        nn.Linear(d_model, n_class),
    )

  def forward(self, graph, input):
    input = self.input_dropout(input)
    input = self.input_embed(input)
    head_output_list = []
    for head in self.head_list:
      head_output = head(graph, input)
      head_output_list.append(head_output)
    
    output = torch.mean(torch.stack(head_output_list, dim=0), dim=0, keepdim=False)
    output = self.top(output)
    return output