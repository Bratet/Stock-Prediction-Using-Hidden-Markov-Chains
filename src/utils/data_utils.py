import yfinance as yf
import numpy as np

def get_stock_prices(company_symbol, start_date, end_date):
    stock_data = yf.download(company_symbol, start=start_date, end=end_date)
    stock_prices = stock_data["Close"].values
    return stock_prices

        
def get_price_movements(stock_prices):
	price_change = stock_prices[1:] - stock_prices[:-1]
	price_movement = np.array(list(map((lambda x: 1 if x>0 else 0), price_change)))
	return price_movement