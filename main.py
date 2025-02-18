import torch
from train import train_rhn, train_dam

if __name__ == '__main__':
    print("Training models...")
    train_rhn()
    train_dam()
