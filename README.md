# Stock Price Movement Prediction Using Hybrid LSTM–Transformer Models

## Overview

This project investigates the use of a Hybrid LSTM–Transformer architecture for predicting next-day stock price movements. The model combines the sequential learning capability of Long Short-Term Memory (LSTM) networks with the self-attention mechanism of Transformers to capture both short-term market dynamics and long-range temporal dependencies.

The study evaluates whether combining recurrent and attention-based architectures improves directional prediction performance compared to traditional statistical and deep learning approaches.

---

## Problem Statement

Financial markets are highly volatile, noisy, and non-stationary, making stock price prediction a challenging task.

The objective of this project is to:

* Predict next-day stock price movements
* Improve directional forecasting accuracy
* Evaluate trading relevance using risk-adjusted metrics
* Compare hybrid deep learning models against classical baselines

---

## Dataset

Historical daily stock market data was collected for:

* Apple Inc. (AAPL)
* Tesla Inc. (TSLA)

Features include:

* Open
* High
* Low
* Close
* Volume (OHLCV)

Additional engineered features include technical indicators and sentiment features.

---

## Feature Engineering

The following technical indicators were incorporated:

### Momentum Indicators

* Relative Strength Index (RSI)
* MACD

### Volatility Indicators

* Bollinger Bands
* Average True Range (ATR)

### Volume Indicators

* On-Balance Volume (OBV)

### Additional Features

* Log Returns
* Volume Transformation
* FinBERT-based Sentiment Scores (optional)

---

## Model Architecture

The proposed architecture consists of:

```text
Input Sequence (60 Days × Features)
        ↓
LSTM Layer (64 Hidden Units)
        ↓
Dropout (0.2)
        ↓
Transformer Encoder
(2 Layers, 4 Attention Heads)
        ↓
Temporal Aggregation
        ↓
Dense Layer
        ↓
Predicted Log Return
```

### Why Hybrid?

* LSTM captures short-term sequential patterns.
* Transformer captures long-range dependencies.
* Combined architecture improves market trend recognition.

---

## Models Evaluated

### Baseline Models

* ARIMA (1,0,1)
* LSTM

### Proposed Model

* Hybrid LSTM–Transformer

---

## Results

| Model              | Directional Accuracy | RMSE    |
| ------------------ | -------------------- | ------- |
| ARIMA              | 59.3%                | 0.15381 |
| LSTM               | 62.8%                | 0.16132 |
| LSTM + Transformer | 65.7%                | 0.07312 |

### Key Findings

* Hybrid LSTM–Transformer achieved the highest directional accuracy.
* Transformer attention improved long-range dependency modelling.
* The model captured market trends more effectively than standalone LSTM.
* Volatility-based features contributed most to predictive performance.

---

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-Learn
* Transformers
* LSTM Networks
* FinBERT
* Financial Time-Series Analysis

---

## Repository Structure

```text
Stock-Price-Movement-Prediction/
│
├── data/
│   └── README.md
│
├── src/
│   └── stock_prediction.ipynb
│
├── outputs/
│   ├── arima_prediction.png
│   ├── lstm_prediction.png
│   ├── hybrid_prediction.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
└── README.md
```

---

## Applications

* Algorithmic Trading
* Portfolio Management
* Risk Management
* Financial Forecasting
* Quantitative Finance Research

---

## Future Improvements

* Real-time News Integration
* Multi-Asset Forecasting
* Reinforcement Learning Trading Agents
* Explainable AI (XAI)
* Cross-Market Transfer Learning

---

## Author

**Veera Suresh Akuthota**
MSc Data Science
University of Roehampton, London

---

## Disclaimer

This project is intended for academic and research purposes only and should not be considered financial advice.
