import torch
from models.rhn import RHN
from models.dam import DAM
from utils.early_stopping import EarlyStopper

# Dummy dataset placeholder
train_images = torch.randn(100, 32, 32)  # 100 random images
train_images = train_images / 255 * 2 - 1
patterns = train_images.reshape(train_images.shape[0], -1)

# Train RHN
rhn = RHN(patterns.shape[1], 15)
early_stopper = EarlyStopper(10, 0, filename='rhn_optimal_weight.pth')
rhn_loss = rhn.loss(patterns, rhn.forward(patterns))
print(f"RHN Initial Loss: {rhn_loss.item()}")

# Train DAM
dam = DAM(n_power=40, m_power=50, k_memories=15)
dam.initialize_memory(patterns.shape[1])
dam.train(patterns, epochs=100)
