import torch
from ogb.nodeproppred import Evaluator

def evaluate(model_factory, data_factory, dataset_name, split_name, metric="acc"):
  evaluator = Evaluator(dataset_name)
  model = model_factory.model

  model.eval()
  graph = data_factory.graph
  y_true = data_factory.labels

  if model.masked:
    pred_idx, x_masked = data_factory.make_mask(split_name)
    logit_pred = model(graph, x_masked)
  else:
    pred_idx = data_factory.get_idx_split(split_name)
    x = graph.ndata["x"]
    logit_pred = model(graph, x)

  y_pred = torch.argmax(logit_pred, dim=1, keepdim=True)
  
  score = evaluator.eval({"y_true":y_true[pred_idx], "y_pred":y_pred[pred_idx]})[metric]

  return score