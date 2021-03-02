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


def evaluate_static(data_factory, dataset_name, split_name, metric="acc"):
  evaluator = Evaluator(dataset_name)

  y_true = data_factory.labels

  pred_idx = data_factory.get_idx_split(split_name)
  train_idx = data_factory.get_idx_split("train")

  y_train_mode, _ = torch.mode(y_true[train_idx], dim=0)
  y_pred = y_train_mode*torch.ones_like(pred_idx).unsqueeze(-1).to(data_factory.device)
  score = evaluator.eval({"y_true":y_true[pred_idx], "y_pred":y_pred})[metric]

  return score