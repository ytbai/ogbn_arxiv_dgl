import torch
from torch import nn

def CrossEntropyLossSmooth(logit_pred, y_true):
   epsilon = 0.3
   output = nn.functional.cross_entropy(logit_pred, y_true.squeeze(1), reduction="none")
   output = smooth(output, epsilon)
   output = torch.mean(output)
   return output

def CrossEntropyLoss(logit_pred, y_true):
   output = nn.functional.cross_entropy(logit_pred, y_true.squeeze(1), reduction="mean")
   return output

def smooth(input, epsilon):
  output = epsilon*torch.log(1+input/epsilon)
  return output