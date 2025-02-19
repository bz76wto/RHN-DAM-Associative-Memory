# RHN-DAM-Associative-Memory

## 📌 Project Overview
This repository implements **Recurrent Highway Networks (RHN)** and **Dense Associative Memory (DAM)** models, inspired by our research paper *"Restricted Hopfield Networks are Robust to Adversarial Attack"*. The focus is on **adversarial robustness and memory retention** through techniques like **Subspace Rotation Algorithm (SRA)** and **energy-based regularization**.

### **Key Features:**
- **📌 SRA Implementation for RHN Training**
  - Enhances weight orthogonality for better stability.
  - Improves memory retrieval robustness under adversarial perturbations.
  - Toggle option for training RHN with/without SRA.
  
- **⚡ Adversarial Attack Evaluation Scripts**
  - Includes **FGSM, PGD, BIM, and Gaussian Noise attacks**.
  - Benchmarks RHN-DAM against other memory models under adversarial stress.
  - Provides visualization of attack impact on retrieval accuracy.
  
- **📝 Structured Repository and Documentation**
  - Step-by-step **setup guide** for training and evaluation.
  - Structured repository with clear separation of models, attacks, and experiments.
  - Jupyter notebooks for interactive testing.
  
- **🏗 Pre-trained Models & Example Datasets**
  - Includes pre-trained RHN-DAM models trained on **alphabet and character datasets**.
  - Provides sample scripts for dataset pre-processing and retrieval evaluation.
  
- **🔬 Energy-Based Loss Function for Robust Memory Networks**
  - Implements energy function regularization for **stable attractor basins**.
  - Visualizes energy landscape across different memory architectures.

## 📂 Repository Structure
```
RHN-DAM-Associative-Memory/
├── README.md
├── models/
│   ├── rhn.py  # Recurrent Highway Network
│   ├── sra.py  # Subspace Rotation Algorithm
├── attacks/
│   ├── fgsm_attack.py  # FGSM Attack
│   ├── pgd_attack.py  # PGD Attack
│   ├── bim_attack.py  # BIM Attack
│   ├── gaussian_noise_attack.py  # Gaussian Noise Attack
│   ├── attack_rhn.py  # Unified script for evaluating attacks
├── experiments/
│   ├── train_rhn_sra.py  # RHN training with SRA
│   ├── eval_robustness.py  # Robustness evaluation script
├── notebooks/
│   ├── visualize_attack_rhn.ipynb  # Interactive visualization
├── datasets/
│   ├── alphabet_data/  # Alphabet dataset for retrieval testing
│   ├── character_data/  # Character dataset for adversarial testing
├── requirements.txt  # Dependencies for installation
```

## ⚙️ Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/bz76wto/RHN-DAM-Associative-Memory.git
cd RHN-DAM-Associative-Memory
```

### **2. Install dependencies**
Ensure you have Python 3.8+ and install required libraries:
```bash
pip install -r requirements.txt
```

### **3. Train RHN-DAM with SRA**
```bash
python experiments/train_rhn_sra.py  # Train RHN with SRA
```

### **4. Evaluate Model Robustness Against Adversarial Attacks**
To test RHN-DAM under different attacks:
```bash
python attacks/attack_rhn.py  # Runs all attacks: FGSM, PGD, BIM, Gaussian Noise
```

For visualization and further analysis:
```bash
jupyter notebook notebooks/visualize_attack_rhn.ipynb
```

## 🧪 Experiments & Results
### **1. Memory Retention Performance**
- RHNs and DAMs evaluated on **synthetic and real-world datasets**.
- Improved **adversarial robustness** over standard memory models.

### **2. Adversarial Robustness Metrics**
- Tested under **FGSM, PGD, BIM, and Gaussian Noise**.
- **SRA-trained RHN** showed **XX% accuracy improvement** over vanilla RHN under attack.

### **3. Energy-Based Training Stability**
- Loss landscapes show **smoother attractor basins** for stable memory retrieval.
- Visualized using `experiments/energy_visualization.py`.

## 📝 Research Context
This repository extends previous research on:
- **Recurrent Highway Networks**: Srivastava et al. (2015) [Paper](https://arxiv.org/abs/1607.03474)
- **Dense Associative Memory**: Krotov & Hopfield (2016) [Paper](https://arxiv.org/abs/1608.06996)
- **Subspace Rotation Training for Robust Memory Networks** (Our Work) [Paper](#)

## 🤝 Contributing Guidelines
We welcome contributions! To contribute:
1. **Fork the repository**
2. **Create a new branch**
3. **Make changes and commit**
4. **Submit a pull request**

For major changes, please open an issue first to discuss proposed modifications.

## 📫 Contact
For questions or collaborations, contact [your email/contact info].

---
This repository is maintained as part of ongoing research on robust memory networks and adaptation strategies in deep learning. Feel free to explore and contribute! 🚀

