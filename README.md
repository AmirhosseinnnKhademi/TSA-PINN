# TSA-PINN

A modular TensorFlow implementation of Physics-Informed Neural Networks (PINNs) intergated with a novel  "Trainable Sinusoidal Activation" applicable to PDEs such as Navier-Stokes equations.

Please review my work published in the Elsevier's journal of computer physics communications:

https://www.sciencedirect.com/science/article/pii/S0010465525001742

DOI: https://doi.org/10.1016/j.cpc.2025.109672

## Repository Structure

```text
TSA-PINN/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
├── configs/
│   └── base.yaml
├── outputs/
├── checkpoints/
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
