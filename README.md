# Probabilistic Digital Twin for Rotating Machinery

## Project Overview
This project is developed for the Probabilistic Machine Learning course. Our goal is to create a probabilistic "digital twin" for fault prediction in industrial rotating machinery (e.g., turbines). We aim to model the system's degradation using Variational Gaussian Processes.

## Dataset
We are using the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset, which contains run-to-failure simulation data for turbofan engines.

## Technologies Used
* **Python**
* **PyTorch & GPyTorch** (for probabilistic modeling)
* **Pandas, NumPy, Scikit-learn** (for data processing and evaluation)

## Repository Structure
pml-digital-twin/
├── data/                   # Store data here
│   ├── raw/                # Raw data downloaded directly from NASA
│   └── processed/          # Data after cleaning and normalization
├── notebooks/              
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   └── 02_svgp_poc.ipynb   # Proof of Concept for the model
├── src/                    # Project source code
│   ├── __init__.py
│   ├── data/               # Scripts for loading and processing data
│   │   ├── __init__.py
│   │   └── data_loader.py  
│   ├── models/             # Model and kernel definitions
│   │   ├── __init__.py
│   │   ├── svgp.py         # Main model class
│   │   └── kernels.py      # Your physics-informed kernels
│   ├── training/           # Training logic and loops
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/              # Helper functions, metrics (NLL, PICP)
│       ├── __init__.py
│       └── metrics.py
├── .gitignore              
├── requirements.txt        
└── README.md               

## References
* Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.
* Zhang, Z. J., et al. (2024). Probabilistic Learning from Real-World Observations of Systems with Unknown Inputs for Model-Form UQ and Digital Twinning.