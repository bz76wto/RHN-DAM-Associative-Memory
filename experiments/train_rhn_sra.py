import torch
import torch.nn as nn
import torch.optim as optim
from models.rhn import RHN  # Importing your RHN model
from data_loader import get_dataset  # Import dataset loading function

# Define Subspace Rotation Algorithm (SRA) for RHN training
def subspace_rotation(W, epsilon=1e-5):
    U, S, V = torch.svd(W)  # Singular Value Decomposition
    S_clamped = torch.clamp(S, min=epsilon)  # Avoid numerical instability
    return torch.mm(U, torch.mm(torch.diag(S_clamped), V.t()))

# RHN Training function with SRA
def train_rhn_sra(model, dataloader, criterion, optimizer, num_epochs=50, use_sra=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply SRA if enabled
            if use_sra:
                for param in model.parameters():
                    if len(param.shape) > 1:  # Apply SRA to weight matrices
                        param.data = subspace_rotation(param.data)
            
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    # Load dataset
    train_loader, test_loader = get_dataset(batch_size=32)
    
    # Initialize RHN model
    model = RHN(input_size=28*28, hidden_size=256, num_layers=3, output_size=10)  # Example config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train RHN with SRA
    trained_model = train_rhn_sra(model, train_loader, criterion, optimizer, num_epochs=50, use_sra=True)
    
    # Save trained model
    torch.save(trained_model.state_dict(), "rhn_sra_model.pth")
