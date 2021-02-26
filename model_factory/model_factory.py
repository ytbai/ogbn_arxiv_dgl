import torch
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

class ModelFactory():
  def __init__(self, model, models_dir, name = None):
    self.model = model
    self.models_dir = models_dir
    self.model_dir = os.path.join(self.models_dir, name)
    self.make_models_dir()
    self.make_model_dir()

    if name is None:
      self.name = self.model.__class__.__name__
    else:
      self.name = name

    self.model_state_dict_path = os.path.join(self.model_dir, "model_state_dict.tar")
    self.optimizer_state_dict_path = os.path.join(self.model_dir, "optimizer_state_dict.tar")
    self.scheduler_state_dict_path = os.path.join(self.model_dir, "scheduler_state_dict.tar")
    self.loss_dict_path = os.path.join(self.model_dir, "loss_dict.p")

    self.loss_dict = defaultdict(list)
    self.mode_dict = defaultdict(str)

  def set_optimizer(self, optimizer):
    self.optimizer = optimizer
  
  def set_scheduler(self, scheduler):
    self.scheduler = scheduler

  def schedule(self):
    return hasattr(self, "scheduler")

  def optimize(self):
    return hasattr(self, "optimizer")

  def set_lr(self, lr):
    for param in self.optimizer.param_groups:
      self.optimizer['lr'] = lr

  def make_models_dir(self):
    if not os.path.exists(self.models_dir):
      os.mkdir(self.models_dir)

  def make_model_dir(self):
    if not os.path.exists(self.model_dir):
      os.mkdir(self.model_dir)

  def save_best(self, loss_name):
    mode = self.mode_dict[loss_name]
    if mode == "min":
      if min(self.loss_dict[loss_name]) == self.loss_dict[loss_name][-1]:
        self.save_model()
    elif mode == "max":
      if max(self.loss_dict[loss_name]) == self.loss_dict[loss_name][-1]:
        self.save_model()
    self.save_loss_dict()
    self.save_optimizer()
    self.save_scheduler()


  def save(self):
    self.save_model()
    self.save_loss_dict()
    self.save_optimizer()
    self.save_scheduler()
  
  def load(self):
    self.load_model()
    self.load_loss_dict()
    self.load_optimizer()
    self.load_scheduler()

  def save_model(self):
    torch.save(self.model.state_dict(), self.model_state_dict_path)
    print("model saved")
  
  def load_model(self):
    self.model.load_state_dict(torch.load(self.model_state_dict_path))
    print("model loaded")

  def save_optimizer(self):
    torch.save(self.optimizer.state_dict(), self.optimizer_state_dict_path)
    print("optimizer saved")
  
  def load_optimizer(self):
    self.optimizer.load_state_dict(torch.load(self.optimizer_state_dict_path))
    print("optimizer loaded")

  def save_scheduler(self):
    if self.schedule():
      torch.save(self.scheduler.state_dict(), self.scheduler_state_dict_path)
      print("scheduler saved")
  
  def load_scheduler(self):
    if self.schedule():
      self.scheduler.load_state_dict(torch.load(self.scheduler_state_dict_path))
      print("scheduler loaded")

  def save_loss_dict(self):
    f = open(self.loss_dict_path, "wb")
    pickle.dump(self.loss_dict, f)
    f.close()
    print("loss_dict saved")
    
  def load_loss_dict(self):
    f = open(self.loss_dict_path, "rb")
    self.loss_dict = pickle.load(f)
    f.close()
    print("loss_dict loaded")

  def add_loss_name(self, loss_name, mode="min"):
    self.loss_dict[loss_name] = []
    self.mode_dict[loss_name] = mode

  def append_loss(self, loss_name, loss_val):
    self.loss_dict[loss_name].append(loss_val)

  def plot_loss_dict(self, include_model_name = False):
    for loss_name in self.loss_dict:
      if include_model_name:
        label = self.name + " " + loss_name
      else:
        label = loss_name
      plt.plot(self.loss_dict[loss_name], label = label)
    plt.legend()


  def print_last_loss(self, epoch = None, verbose = True):
    if verbose == False:
      return
    if epoch is not None:
      print("epoch: %d" % epoch, end = " ")

    for key in self.loss_dict:
      print("%s %f" % (key, self.loss_dict[key][-1]), end = " ")

    print("")

  def print_best_loss(self):
    for key in self.loss_dict:
      if self.mode_dict[key]=="min":
        best_loss = min(self.loss_dict[key])
      elif self.mode_dict[key]=="max":
        best_loss = max(self.loss_dict[key])

      print("best %s %f" % (key, best_loss), end = " ")
    print(" ")

  def get_num_params(self, requires_grad_only):
    if requires_grad_only:
      return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    else:
      return sum(p.numel() for p in self.model.parameters())

  def print_num_params(self):
    num_params_total = self.get_num_params(requires_grad_only=False)
    num_params_requires_grad = self.get_num_params(requires_grad_only=True)
    num_params_no_grad = num_params_total-num_params_requires_grad
    print("Number of parameters (total): %d" % num_params_total)
    print("Number of parameters (requires grad): %d" % num_params_requires_grad)
    print("Number of parameters (no grad): %d" % num_params_no_grad)