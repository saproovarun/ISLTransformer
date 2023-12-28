import torch
from torch.nn import CrossEntropyLoss

class AccuracyTrack:
    def __init__(self):
        self.corrections = 0
        self.total_examples = 0
    def update(self, preds, target):
        self.corrections += torch.sum(torch.argmax(preds, dim=1) == target)
        self.total_examples += preds.shape[0]
    def item(self):
        return (100*self.corrections/self.total_examples).item()
    def reset(self):
        self.corrections = 0
        self.total_examples = 0

class LossTrack:
    def __init__(self, criterian):
        self.total_loss_val = 0
        self.steps = 0
        self.eval_func = criterian()
    def update(self, preds, target):
        self.total_loss_val += self.eval_func(preds, target)
        self.steps += 1
    def item(self):
        return (self.total_loss_val / self.steps).item()
    def reset(self):
        self.total_loss_val = 0
        self.steps = 0