# TSA-PINN

A modular TensorFlow implementation of **Physics-Informed Neural Networks (PINNs)** with **Trainable Sinusoidal Activation (TSA)** for solving nonlinear PDEs such as the **Navier–Stokes equations**.

This repository reproduces and demonstrates the **TSA-PINN architecture** proposed in our research for improving convergence and accuracy in physics-informed deep learning models.

---

## 📄 Associated Publication

Please review the research paper published in **Elsevier – Computer Physics Communications**:

Paper:  
https://www.sciencedirect.com/science/article/pii/S0010465525001742

DOI:  
https://doi.org/10.1016/j.cpc.2025.109672

If you use this work, please cite the paper.

---

# Overview

Physics-Informed Neural Networks (PINNs) incorporate physical laws (PDEs) directly into the loss function of neural networks. However, standard PINNs may suffer from:

- slow convergence
- spectral bias
- difficulty capturing high-sigma dynamics

To address these challenges, we propose **TSA-PINN**, which introduces **trainable sinusoidal activations** into the neural network architecture.

This allows the model to **adaptively learn relevant sigma components** of the solution.

---

# Project Structure

```text
TSA-PINN/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── colab_runner.ipynb
├── configs/
│   └── standard.yaml
│   └── tsa_sigma_0_1.yaml
│   └── tsa_sigma_1_0.yaml
│   └── tsa_sigma_3_0.yaml
├── outputs/
│   └── standard.yaml
│   └── tsa_sigma_0_1.yaml
│   └── tsa_sigma_1_0.yaml
│   └── tsa_sigma_3_0.yaml
├── checkpoints/
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
