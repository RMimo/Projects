# Projects
## [End-to-End Workflow for House Price Prediction Using Machine Learning](https://colab.research.google.com/drive/1jTEhWglRgdC5eMfFgaKAMcNDoV2HfqDe?usp=sharing)
This notebook demonstrates the end-to-end process of building, training, and deploying a fully-connected neural network for predicting house prices. The workflow begins by loading the dataset from Kaggle, followed by essential data preprocessing and feature generation, including normalization, bucketization, and one-hot encoding. The dataset is then split into training and testing sets. A sequential neural network model is constructed, trained with early stopping, and exported for future use. After loading the trained model, predictions are made on new data points. Finally, the model is deployed as a web application, allowing users to input features and receive price predictions via a user-friendly interface hosted on a cloud service (Hugging Face Spaces).


**Keywords**: Data Preprocessing, Feature Engineering, Model Training, Model Deployment, Neural Network, Price Prediction, End-to-End Machine Learning Pipeline 
**Libraries**: tensorflow.keras, sklearn, pandas, numpy, pickle, json, matplotlib, streamlit

<!--- 

Quantization, RAG 
- Vector Database Management: Set up and manage vector databases

Quantization, Fine-tuning
- Evaluate

Image

--->

## [Earning Call Report Generation using Large Language Models](https://colab.research.google.com/drive/1qhf6yukdh87epegSNZ5fo8-vStrEyQew?usp=sharing)
For investors seeking to gain an edge in the market, the ability to efficiently analyze the content of earnings calls is crucial. Large language models (LLMs) offer a powerful solution to the challenges posed by unstructured data, enabling investors to extract meaningful insights and make better-informed decisions. This notebook outlines the development of a pipeline to analyze financial YouTube videos. The process involves obtaining and preprocessing the video transcript, extracting key concepts, filtering the content, and performing sentiment analysis using FinBERT. The transcript is then summarized with the PEGASUS for Financial Summarization model, a final report and a list of key takeaways are generated using Microsoft Phi-3-mini-4k-instruct model. This pipeline is specifically demonstrated on the Alphabet Q2 2024 earnings call.

**Keywords**: Unstructured Data, Large Language Models, Earning Calls, Sentiment Analysis, Text Summarization
**Libraries**: transformers, nltk, torch, plotly, re, collections, textwrap, youtube_transcript_api, pandas, numpy, matplotlib, tqdm

## [Long-only Trading Strategy based on Random Forests to Forecast Stock Movements](https://colab.research.google.com/drive/1_QqpvhfTRzi7BVRl8kMlCPM47qMjK_c2?usp=sharing)
In this project, we create a long-only trading strategy based on the application of Random Forests to predict stock market trends. The project begins with an introduction to Random Forests, detailing how they operate, their advantages, and limitations. It then delves into the critical aspect of hyperparameter tuning, presenting common hyperparameters tuning techniques like Grid Search, Random Search, and Bayesian Optimization. The section on feature engineering highlights its benefits and provides examples, including feature creation and transformation. The implementation phase involves loading libraries and data, followed by detailed feature engineering techniques such as Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD). The data is then split into training, validation, and test sets. We establish several baselines, including risk-free rate, buy and hold, random strategy, and S&P 500 index, to benchmark our strategy. After hyperparameter tuning, we backtest our strategy based on the models that provided the highest Sharpe ratio, maximum returns, lowest volatility, and highest accuracy. Finally, the results are presented and compared against the baselines. Additionally, feature importance is presented, showing the contributions of various features to the predictions of the highest return model.


**Keywords**: Random Forests, Hyperparameters Tuning, Feature Engineering, Relative Strength Index, Moving Average Convergence Divergence, Backtesting Baselines 
**Libraries**: yfinance, pandas, numpy, matplotlib, tqdm, sklearn

## [Pairs Trading GOOG-GOOGL](https://colab.research.google.com/drive/1_tW2j2ZyMuuAOcp6oOKhXqXySO42FsgI?usp=sharing)
This notebook provides a comprehensive exploration of pairs trading, a popular statistical arbitrage strategy, focusing on its mathematical foundations, implementation, and optimization. We start with an introduction to pairs trading and its mathematical underpinnings, including the concept of the Z-score and its application in trading strategies. The notebook details a simple process for optimizing exit and entry thresholds to enhance trading performance. We use S&P500 performance as a baseline and proceed with the practical implementation of the strategy. Key steps include the Engle-Granger cointegration test to identify cointegrated pairs and the visualization of optimal exit and entry thresholds. The backtesting section evaluates two different strategies: trading strategy optimized for the highest total return and optimized for highest sharpe ratio. Performance is assessed on both validation and test sets, providing insights into the robustness and effectiveness of the strategies.


**Keywords**: Pairs Trading, Cointegration, Backtesting 
**Libraries**: yfinance, pandas, numpy, matplotlib, tqdm, statsmodels

<!-- [Stock Movement Prediction using LSTM](https://colab.research.google.com/drive/1H_Dn58foWjGl6U-fi8yoTpL3z3clVI9R?usp=sharing)
This notebook guides you through using Long Short-Term Memory (LSTM) neural networks to predict stock movements. It starts with an introduction to the core concepts, including what LSTMs are, how they are applied, their advantages and limitations, and their main components (gates). Next, it introduces the concept of backtesting, covering its limitations and important metrics such as **total return, annualized return, annualized standard deviation, drawdown, sharpe ratio, and win/loss ratio**. 

The implementation section walks you through the practical steps: importing libraries, collecting and preprocessing data, calculating returns, scaling data, creating sequences, and splitting these sequences into training, validation, and test sets. It also covers building and training the LSTM model using Dropout and Batch Normalization, making returns predictions, and evaluating the accuracy of predictions for upward and downward moves. A simple, long-only trading strategy is built based on the LSTM predictions to demonstrate the concept of **backtesting**. The notebook concludes conducting an evaluation of the performance of the strategy using the metrics mentioned above.
 

**Keywords**: Long Short-Term Memory (LSTM) neural networks, Backtesting, Portfolio Metrics, Trading Strategy 
**Libraries**: yfinance, pandas, numpy, matplotlib, seaborn, sklearn, tensorflow, tensorflow.keras, random, os
-->

## [Black-Scholes Model for European Option Pricing](https://colab.research.google.com/drive/1a580QGd6wzIviVfXvYnLDacntJ5-V-uc?usp=sharing)
This notebook explores the fundamentals of the **Black-Scholes option pricing model**. It begins with an overview of key concepts, including the Black-Scholes formula and the assumptions underpinning the model. The notebook then delves into the **Black-Scholes Greeks**, which are essential for understanding how different factors affect options pricing. These Greeks include Delta (Δ), Gamma (Γ), Vega (ν), Theta (Θ), and Rho (ρ). Finally, the notebook includes an example and sanity check to illustrate the practical application of the model and verify its calculations. 


**Keywords**: Black-Scholes Model, Black-Scholes Greeks 
**Libraries**: numpy, scipy.stats, py_vollib

## [Calculating Option Implied Volatility using Newton-Raphson Method](https://colab.research.google.com/drive/17bx4AO06UQs0qWrbYsx2LouKfNs_V_ek?usp=sharing)
This notebook provides a thorough exploration of implied volatility and its significance in financial modeling. It starts with an introduction to the basic concept of volatility and explains why implied volatility is a crucial measure. The notebook then covers how to find implied volatility using the Newton-Raphson method. The notebook outlines the steps involved in the Newton-Raphson method, including its advantages and disadvantages. The process of calculating implied volatility is detailed, formulating the problem and applying the Newton-Raphson iterative formula with specific focus on the importance and application of the Black-Scholes formula in this method. The notebook also presents the iterative algorithm used for calculations, culminating in a practical implementation of the method.


**Keywords**: Implied Volatility, Newton-Raphson Method, Black-Scholes Model 
**Libraries**: numpy, scipy.stats

## [Calculating VaR and CVaR using Monte Carlo Simulations](https://colab.research.google.com/drive/1aBb-kQydSxyHwIVK63dFNDraU9WALMTN?usp=sharing) 
This notebook delves into the principles and applications of Monte Carlo simulation. It begins with key concepts and outlines the essential steps involved in running a Monte Carlo simulation. The notebook includes a hands-on example for estimating the value of Pi, accompanied by visualizations to illustrate the simulation process. Expanding on these fundamentals, the notebook applies Monte Carlo methods to stock portfolio analysis, focusing on Value-at-Risk (VaR) and its key components. The notebook also introduces **Conditional Value-at-Risk (CVaR)** as a complementary risk of VaR assessment tool. 

**Keywords**: Monte Carlo Simulations, Value-at-Risk, Conditional Value-at-Risk 
**Libraries**: yfinance, pandas, numpy, datetime, random, matplotlib

## [Financial Data Analysis and Visualization](https://colab.research.google.com/drive/1JYAsR-_XC_9HAg7yOc9LiiVl9a07-zYQ?usp=sharing)
This notebook is a simple step-by-step to analyze and visualize financial data. It covers importing necessary libraries, collecting stock data from Yahoo Finance, and exploring and cleaning the data. Key metrics such as closing prices, moving averages, trading volumes, and Bollinger Bands are visualized. Additionally, the notebook includes analysis of financial metrics such as daily and cumulative returns and correlation analysis to uncover relationships between different financial indicators.


**Keywords**: Data Visualization, Financial Metrics 
**Libraries**: yfinance, pandas, numpy, matplotlib, seaborn

<!--
## [XGBoost]()
---


## [PCA]()
---


## [ARIMA]()
---


## [GARCH]()
---

-->
