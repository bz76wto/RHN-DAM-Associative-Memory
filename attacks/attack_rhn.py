import torch
import torch.nn.functional as F
from models.rhn import RHN  # Import RHN model
from data_loader import get_dataset  # Import dataset loader

# FGSM Adversarial Attack
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed_images = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed_images, 0, 1)

# PGD Adversarial Attack
def pgd_attack(model, images, labels, epsilon, alpha, iters):
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True
    
    for _ in range(iters):
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        perturbed_images = perturbed_images + alpha * images.grad.sign()
        perturbed_images = torch.min(torch.max(perturbed_images, images - epsilon), images + epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()
        perturbed_images.requires_grad = True
    
    return perturbed_images

# BIM Adversarial Attack
def bim_attack(model, images, labels, epsilon, alpha, iters):
    return pgd_attack(model, images, labels, epsilon, alpha, iters)

# Gaussian Noise Attack
def gaussian_noise_attack(images, mean=0, std=0.1):
    noise = torch.randn_like(images) * std + mean
    return torch.clamp(images + noise, 0, 1)

# Evaluate model under attacks
def evaluate_under_attack(model, dataloader, attack_fn, attack_name, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        perturbed_images = attack_fn(model, images, labels, **kwargs) if 'labels' in attack_fn.__code__.co_varnames else attack_fn(images, **kwargs)
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total
    print(f'Adversarial Accuracy under {attack_name}: {accuracy:.4f}')
    return accuracy

if __name__ == "__main__":
    # Load dataset
    _, test_loader = get_dataset(batch_size=32)
    
    # Load trained RHN model
    model = RHN(input_size=28*28, hidden_size=256, num_layers=3, output_size=10)
    model.load_state_dict(torch.load("rhn_sra_model.pth"))
    
    # Evaluate model under different attacks
    epsilons = [0.05, 0.1, 0.2]
    alpha = 0.01
    iters = 10
    
    for eps in epsilons:
        evaluate_under_attack(model, test_loader, fgsm_attack, "FGSM", epsilon=eps)
        evaluate_under_attack(model, test_loader, pgd_attack, "PGD", epsilon=eps, alpha=alpha, iters=iters)
        evaluate_under_attack(model, test_loader, bim_attack, "BIM", epsilon=eps, alpha=alpha, iters=iters)
    
    evaluate_under_attack(model, test_loader, gaussian_noise_attack, "Gaussian Noise", mean=0, std=0.1)
