# Market-Context Stock Transformer (MCST)

This repository contains the code for the Market-Context Stock Transformer (MCST) project. The MCST is a Transformer-based model designed to forecast next-day normalised stock returns by fusing stock-specific data with broader market context. The project was developed as part of COMP0162 module at UCL.

## Overview

The MCST model leverages:
- **Dual Encoder Architecture:**  
  Two independent Transformer encoders—one for stock-specific features (e.g., OHLC prices and trading volume) and one for market-specific features (e.g., SPX, VIX, COR1M).
- **Fusion Module:**  
  A fusion layer (using cross-attention, as demonstrated in ablation studies) that integrates outputs from both encoders.
- **Prediction Head:**  
  An attention-based or last time-step aggregation strategy that outputs the next day’s normalised return.

According to the project report, the model significantly outperforms benchmarks, including a comparable LSTM model, achieving high directional accuracy and lower error rates even when trained on a modest dataset (~500k samples).

## Repository Structure

```
COMP0162-project/
├── data/                     # Raw and processed datasets
├── data_processing/          # Scripts for cleaning and preparing data
├── model/                    # Model definitions (MCST architecture)
├── training/                 # Training scripts for the MCST model
├── testing/                  # Scripts for model evaluation
├── output/                   # Directory for storing outputs and results
├── remote_ablation.py        # Script to run ablation studies
├── remote_lstm_training.py   # Script to train an LSTM baseline model
├── remote_processing.py      # Script for remote data processing
├── remote_training.py        # Script to train the MCST model remotely
├── requirements.txt          # General Python package dependencies
└── torch_requirements.txt    # PyTorch-specific dependencies
```

## Getting Started

### Prerequisites

- **Python:** 3.8 or later
- **PyTorch:** Refer to `torch_requirements.txt` for specific version requirements
- Other dependencies can be installed from `requirements.txt`

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ben256/COMP0162-project.git
cd COMP0162-project
pip install -r requirements.txt
pip install -r torch_requirements.txt
```

### Data Preparation
- Use the scripts in the `data_processing/` directory to clean and prepare your data according to the preprocessing steps outlined in the report.
- Market data is already on GitHub, but stock data needs to be downloaded from Yahoo Finance. The `data_processing/process_stock_data.py` script can be used to do this.
- `remote_processing` should perform all the necessary data processing steps in one, and provides customisable option save locations.

### Training the Model

To train the MCST model:

```bash
python remote_training.py
```

For ablation studies to compare different fusion and prediction strategies:

```bash
python remote_ablation.py
```

For training an LSTM baseline (as per the report’s benchmark):

```bash
python remote_lstm_training.py
```

### Evaluation

Once training is complete, evaluation scripts located in the `testing/` directory can be used to assess model performance and generate results.

## Project Details

The MCST model integrates stock-specific historical data with market context to improve prediction accuracy. Key highlights include:

- **Architecture:**
    - Stock encoder processes a 22-dimensional input (OHLC and volume data).
    - Market encoder processes a 24-dimensional input (market indicators).
- **Fusion Module:**
    - Cross-attention is used to merge the two modalities, leading to richer representations.
- **Prediction Head:**
    - Two approaches were compared: using the final time-step representation vs. attention pooling over all time steps.
- **Results:**
    - Ablation studies in the report show that cross-attention fusion and attention pooling provide significant performance improvements over simpler methods.

For a complete explanation of the methodology, experimental setup, and results, please refer to the [full project report]().

---

For any questions or issues, please open an issue on GitHub or contact Benjamin Naylor at [benjamin.naylor.24@ucl.ac.uk](mailto:benjamin.naylor.24@ucl.ac.uk).