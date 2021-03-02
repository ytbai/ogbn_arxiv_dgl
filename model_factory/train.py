import torch
from torch import nn
import numpy as np

def train(model_factory, data_factory, criterion):
  model = model_factory.model
  optimizer = model_factory.optimizer
  scheduler = model_factory.scheduler

  model.train()

  optimizer.zero_grad()

  graph = data_factory.graph
  y_true = data_factory.labels

  if model.masked:
    pred_idx, x_masked = data_factory.make_mask("train")

    logit_pred = model(graph, x_masked)

  else:
    pred_idx = data_factory.get_idx_split("train")
    x = graph.ndata["x"]
    logit_pred = model(graph, x)

  loss = criterion(logit_pred[pred_idx], y_true[pred_idx])
  loss.backward()
  optimizer.step()

  if model_factory.schedule():
    scheduler.step(loss.item())

  return loss.item()