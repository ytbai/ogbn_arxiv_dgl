import torch
from torch import nn
import dgl
import dgl.function as fn

class GATConv(nn.Module):
  def __init__(
    self,
    d_input,
    d_head,
    n_head,
    include_top,
    average_heads,
    include_bias,
  ):

    super().__init__()
    self.d_input = d_input
    self.d_head = d_head
    self.n_head = n_head
    self.include_top = include_top
    self.average_heads = average_heads
    self.include_bias = include_bias

    self.negative_slope = 0.2

    self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    self.input_embed = nn.Linear(d_input, d_head*n_head, bias=False)
    self.a_src = nn.Parameter(torch.zeros(1, n_head, d_head))
    self.a_dst = nn.Parameter(torch.zeros(1, n_head, d_head))

    # linear layer is applied after averaging heads
    if average_heads:
      self.linear = nn.Linear(self.d_input, d_head, bias=False)
    else:
      self.linear = nn.Linear(self.d_input, d_head*n_head, bias=False)
    

    if include_top:
      self.top_drop_rate = 0.75
      self.top = nn.Sequential(
          nn.BatchNorm1d(n_head*d_head),
          nn.ReLU(),
          nn.Dropout(self.top_drop_rate)
      ) 
    
    if include_bias:
      self.bias = nn.Parameter(torch.zeros(1, n_head*d_head))

    self.reset_parameters()

  def reset_parameters(self):
    gain = nn.init.calculate_gain("relu")
    nn.init.xavier_normal_(self.input_embed.weight, gain=gain)
    nn.init.xavier_normal_(self.linear.weight, gain=gain)
    nn.init.xavier_normal_(self.a_src, gain=gain)
    nn.init.xavier_normal_(self.a_dst, gain=gain)

    if self.include_bias:
      nn.init.zeros_(self.bias)

  def gatconv(self, graph, x):
    with graph.local_scope():
      h = self.input_embed(x).view(-1, self.n_head, self.d_head)

      dot_src = torch.sum(h*self.a_src, dim=-1, keepdim=True)
      dot_dst = torch.sum(h*self.a_dst, dim=-1, keepdim=True)

      graph.ndata["h"] = h
      graph.srcdata["dot_src"] = dot_src
      graph.dstdata["dot_dst"] = dot_dst

      graph.apply_edges(dgl.function.u_add_v("dot_src", "dot_dst", "dot"))
      logit = self.leaky_relu(graph.edata["dot"])
      graph.edata["prob"] = dgl.ops.edge_softmax(graph, logit)

      graph.update_all(fn.u_mul_e("h", "prob", "h_prob"), fn.sum("h_prob", "output"))
      output = graph.ndata["output"]

      return output

  def forward(self, graph, x):
    output = self.gatconv(graph, x)

    if self.average_heads:
      output = output.view(-1, self.n_head, self.d_head)
      output = torch.mean(output, dim=1, keepdim=False)
    else:
      output = output.flatten(1)
    
    output += self.linear(x)

    if self.include_top:
      output = self.top(output)

    if self.include_bias:
      output += self.bias

    return output


class GAT(nn.Module):
  def __init__(
    self,
    d_input,
    n_class,
    d_head,
    n_layer,
    n_head,
    masked,
  ):
    assert n_layer >= 3

    super().__init__()
    self.d_input = d_input
    self.n_class = n_class
    self.d_head = d_head
    self.n_layer = n_layer
    self.n_head = n_head
    self.masked = masked
    self.input_drop_rate = 0.1

    self.input_drop = nn.Dropout(self.input_drop_rate)

    self.layers = nn.ModuleList()

    self.layers.append(GATConv(d_input, d_head, n_head, include_top=True, average_heads=False, include_bias=False))
    self.layers.extend([
                        GATConv(d_head*n_head, d_head, n_head, include_top=True, average_heads=False, include_bias=False)
                        for _ in range(n_layer-2)
    ])

    self.layers.append(GATConv(d_head*n_head, n_class, n_head=1, include_top=False, average_heads=True, include_bias=True))


  def forward(self, graph, x):
    h = self.input_drop(x)

    for layer in self.layers:
        h = layer(graph, h)

    return h
