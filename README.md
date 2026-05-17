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
* `/data` - Contains the C-MAPSS dataset files
* `/notebooks` - Jupyter notebooks for exploratory data analysis, plotting, and baseline testing.
* `/src` - Source code for the SVGP models, physics-informed kernels, and online updating logic.
* `requirements.txt` - Python dependencies required to run the project.

## References
* Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.
* Zhang, Z. J., et al. (2024). Probabilistic Learning from Real-World Observations of Systems with Unknown Inputs for Model-Form UQ and Digital Twinning.