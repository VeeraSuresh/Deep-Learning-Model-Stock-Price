# Deep-Learning-Model-Stock-Price
Stock Price Movement Prediction using Hybrid LSTM–Transformer

This repository contains the code and experiments for a research project focused on predicting short-term stock price movements using a hybrid deep learning architecture that combines Long Short-Term Memory (LSTM) networks with Transformer-based self-attention. The project evaluates whether hybrid sequential models can improve directional predictability and risk-adjusted performance compared to classical statistical and standalone deep learning baselines.

🔍 Problem Statement

Financial markets are highly noisy, nonlinear, and non-stationary, making accurate short-horizon stock price prediction extremely challenging. Traditional statistical models struggle to capture complex temporal dependencies, while pure deep learning models often overfit noise. This project investigates whether a hybrid LSTM–Transformer architecture can better capture meaningful temporal structure and improve price movement (up/down) prediction, even when exact return forecasting remains difficult.

🎯 Objectives

Predict daily stock price movements (UP/DOWN) using historical market data

Compare classical (ARIMA) and deep learning baselines (LSTM)

Propose and evaluate a hybrid LSTM–Transformer model

Analyse model confidence using conviction-based filtering

Evaluate performance using both statistical and trading-oriented metrics

📊 Dataset

Primary asset: Apple Inc. (AAPL)

Frequency: Daily data

Features:

OHLCV (Open, High, Low, Close, Volume)

Technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV)

News sentiment (FinBERT-based, limited due to API constraints)

⚠️ Note: Due to restricted access to real-time news APIs, sentiment data coverage is limited. As a result, price- and volatility-based features dominate model performance.

🧠 Model Architectures
1. ARIMA (Baseline)

A classical statistical time-series model used as a linear benchmark.

2. LSTM (Deep Learning Baseline)

A recurrent neural network that captures short-term temporal dependencies in historical price sequences.

3. Hybrid LSTM–Transformer (Proposed Model)

LSTM layer for local sequential pattern learning

Transformer encoder for long-range temporal attention

Dropout and regularisation to prevent overfitting

Final output predicts log return, converted into price movement direction

This hybrid design enables the model to focus on informative historical periods while remaining robust to noise.

⚙️ Training Strategy

Optimizer: Adam with L2 regularisation (weight decay = 1e-4)

Learning Rate Scheduler: StepLR (learning rate reduced by 50% every 10 epochs)

Early Stopping: Training stops if validation loss does not improve for 10 epochs

Batch Size: 32

Sequence Length: 60 trading days

📈 Evaluation Metrics

The model is evaluated using both prediction accuracy and trading relevance metrics:

RMSE / MAE / R² – numerical prediction quality

Directional Accuracy – correctness of UP/DOWN predictions

Sharpe Ratio (daily) – risk-adjusted performance of a strategy following model signals

Conviction Metrics – performance on high-confidence predictions only

Permutation Feature Importance – interpretability and sensitivity analysis

🧪 Key Results (Summary)

Hybrid LSTM–Transformer outperforms ARIMA and LSTM baselines in directional accuracy

High-conviction trades achieve significantly higher accuracy than global predictions

Price- and volatility-based indicators are the most influential features

Positive Sharpe ratio indicates economically meaningful signals despite market noise

📂 Repository Structure
├── data/
│   ├── stock_price_data.csv
│   └── (optional) news_data.csv
├── models/
│   ├── lstm_model.py
│   └── hybrid_lstm_transformer.py
├── experiments/
│   ├── training_pipeline.py
│   ├── evaluation.py
│   └── ablation_studies.py
├── results/
│   ├── plots/
│   └── Research_Features_and_Predictions.csv
├── README.md
└── requirements.txt

🚀 How to Run

Install dependencies:

pip install -r requirements.txt


Prepare data:

Place stock price CSV in /data

(Optional) Add news data if available

Train and evaluate:

python experiments/training_pipeline.py

⚠️ Limitations

Evaluated primarily on a single stock (AAPL)

Daily frequency only

No transaction costs or slippage included

Limited sentiment data due to API constraints

Performance may degrade under extreme out-of-distribution market events

🔮 Future Work

Extend to multi-asset and cross-sectional modelling

Integrate reliable real-time news or alternative data

Add uncertainty-aware or probabilistic prediction heads

Incorporate transaction costs and portfolio optimisation
