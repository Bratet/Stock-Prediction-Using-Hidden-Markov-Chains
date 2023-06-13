# Hidden Markov Model for Stock Price Movements

This project implements a Hidden Markov Model (HMM) to model stock price movements. The model is trained using the Baum-Welch algorithm and makes predictions using the Viterbi algorithm. The model predicts whether the stock price will rise or fall in the following trading day.

## Overview

This project uses the Yahoo Finance API to retrieve historical stock prices for a given company. The prices are transformed into a binary format where a 1 represents an increase in price from the previous day, and a 0 represents a decrease.

The Hidden Markov Model is then trained on this sequence of binary values. Once the model is trained, it can be used to predict future price movements.

## Dataset

The project retrieves stock price data from the Yahoo Finance API. The user can specify the company symbol (for example, "AAPL" for Apple Inc.), and the date range for the historical data.

## Libraries Used

- numpy
- tqdm
- yfinance
- matplotlib

## Usage

To train the model, run the following command:

```python
python main.py
```

After training the model, you can use it to predict future price movements:

```python
python predict.py
```

## Results

The model's accuracy is evaluated by comparing the predicted price movements with the actual price movements in a test dataset. 

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

This project was inspired by the paper "Hidden Markov Models for Financial Forecasting" by Ephraim, Malah and Sondhi.

---

Please modify this README to fit the exact details and requirements of your project.
