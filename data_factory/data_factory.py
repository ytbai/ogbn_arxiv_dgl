import torch
from torch import nn
from ogb.nodeproppred import DglNodePropPredDataset
import dgl


class DataFactory():
  def __init__(self, dataset_name, device, root, include_year=False):
    self.dataset_name = dataset_name
    self.device = device
    self.root = root
    self.include_year = include_year
    self.mask_rate = 0.5

    self.dataset = DglNodePropPredDataset(self.dataset_name, self.root)
    self.graph, self.labels = self.dataset[0]
    self.n_class = self.labels.max().item()+1

    self.make_node_features()

    self.make_bidirected()
    self.add_self_loops()

    self.to_device()

  def to_device(self):
    self.graph = self.graph.to(self.device)
    self.labels = self.labels.to(self.device)

  def make_node_features(self):
    if self.include_year:
     # make year features
      year = self.graph.ndata["year"]-self.graph.ndata["year"].min()
      self.n_year = year.max()+1
      year_one_hot = nn.functional.one_hot(year.squeeze(1), self.n_year)
      
      # concatenate all features
      self.graph.ndata["x"] = torch.cat([self.graph.ndata["feat"], year_one_hot], dim=1)
    else:
      self.graph.ndata["x"] = self.graph.ndata["feat"]

    

  def add_self_loops(self):
    self.graph = dgl.add_self_loop(self.graph)

  def make_bidirected(self):
    self.graph = dgl.to_bidirected(self.graph, copy_ndata=True)

  def get_idx_split(self, split_name):
    return self.dataset.get_idx_split()[split_name]
  
  def make_mask(self, split_name):
    # make mask
    train_idx = self.get_idx_split("train")

    if split_name == "train":
      mask = torch.rand(train_idx.shape) < self.mask_rate # =1 on mask
      preserved_idx = train_idx[~mask] # indices for nodes whose label features are preserved
      pred_idx = train_idx[mask] # indices for nodes whose label features need to be predicted
    elif split_name in ["valid", "test"]:
      preserved_idx = train_idx
      pred_idx = self.get_idx_split(split_name)

    label_feat = torch.zeros([self.graph.num_nodes(), self.n_class]).to(self.device)
    label_feat[preserved_idx, self.labels[preserved_idx, 0]] = 1
    x_masked = torch.cat([self.graph.ndata["x"], label_feat], dim=1)
    return pred_idx, x_masked
