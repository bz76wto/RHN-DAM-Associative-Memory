import torch
import numpy as np

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, filename='optimal_weight.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf
        self.filename = filename

    def early_stop(self, model, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            torch.save(model.w_xh, self.filename)
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                model.w_xh = torch.load(self.filename)
                return True
        return False
