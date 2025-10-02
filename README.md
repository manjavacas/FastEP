# FastEP ⚡️🏢

Surrogate EnergyPlus models for fast building energy optimization.

## 🚀 Overview

FastEP is a project focused on fast building energy optimization. Its goal is to train surrogate models based on EnergyPlus simulations. FastEP enables the generation, training, and evaluation of models that accurately simulate building energy behavior, significantly accelerating optimization and analysis without the need to run full EnergyPlus simulations.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/manjavacas/fastep.git
   cd fastep
   ```

2. Set up your virtual environment with `uv`:
   ```bash
   pip install uv
   uv venv
   source .venv/bin/activate
   ```

3. Install dependencies with `uv` (Python 3.12+ required):
   ```bash
   uv sync
   ```

## 📊 Data Processing

1. **Prepare datasets**. Process raw EnergyPlus simulation data:
   ```bash
   python fastep/src/create_dfs.py
   ```

2. **Train models**. Train LSTM surrogate model:
   ```bash
   python fastep/src/main.py
   ```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.