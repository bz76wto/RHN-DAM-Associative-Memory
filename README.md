# RHN-DAM-AssociativeMemory

## Description
This repository implements **Recurrent Highway Networks (RHN)** and **Dense Associative Memory (DAM)** using PyTorch for associative memory tasks. The models are designed to store and retrieve patterns from memory, similar to Hopfield Networks but with enhancements.

---

## **Installation**
Clone the repository and install dependencies:
```bash
git clone https://github.com/bz76wto/RHN-DAM-AssociativeMemory.git
cd RHN-DAM-AssociativeMemory
pip install -r requirements.txt
```
---

## **Usage**
### **Training RHN and DAM Models**
Run the training script:
```bash
python train.py
```
This script:
- Loads and preprocesses images
- Trains both **RHN** and **DAM** models
- Saves the best model weights

---

## **Project Structure**
```
RHN-DAM-AssociativeMemory/
│── models/
│   ├── rhn.py                # Recurrent Highway Network implementation
│   ├── dam.py                # Dense Associative Memory implementation
│── data/
│   ├── test_images.pt        # Sample dataset
│── utils/
│   ├── early_stopping.py     # Early stopping mechanism
│── train.py                  # Training script for both RHN and DAM
│── main.py                   # Entry point to run training and evaluation
│── requirements.txt          # Dependencies
│── README.md                 # Project overview and instructions
│── LICENSE                   # License file
│── .gitignore                # Ignore unnecessary files (e.g., __pycache__)
```

---

## **Models**
### **Recurrent Highway Network (RHN)**
- A recurrent model that maps inputs into a subspace using **tanh activation**.
- Optimized using **Singular Value Decomposition (SVD)**.
- Uses **early stopping** to prevent overfitting.

### **Dense Associative Memory (DAM)**
- Stores patterns in a weight matrix (`weight_memory`).
- Uses **power function-based updates** to reinforce correct recall.
- Weight updates include **learning rate decay** for stability.

---

## **Results Visualization**
To visualize results:
```bash
python main.py
```
This script will plot **original vs. retrieved patterns**.

---

## **License**
This project is licensed under the MIT License.

---

## **Contributing**
Feel free to submit issues and pull requests!

---

## **Contact**
For questions, email **bzhan138@uottawa.ca**
