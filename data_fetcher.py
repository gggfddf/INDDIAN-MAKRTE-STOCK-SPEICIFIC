import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from nsepy import get_history
import warnings
warnings.filterwarnings('ignore')

class IndianStockDataFetcher:
    """
    Comprehensive data fetcher for Indian stock market with multiple sources
    """
    
    def __init__(self, symbol="RELIANCE.NS"):
        self.symbol = symbol
        self.nse_symbol = symbol.replace('.NS', '')
        self.data_cache = {}
        
    def fetch_yfinance_data(self, period="max", interval="1d"):
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            if not data.empty:
                data.index = pd.to_datetime(data.index)
                return data
        except Exception as e:
            print(f"YFinance error for {interval}: {e}")
        return None
    
    def fetch_nse_data(self, start_date=None, end_date=None):
        """Fetch data from NSE using nsepy"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365*5)  # 5 years
            if end_date is None:
                end_date = datetime.now()
                
            data = get_history(
                symbol=self.nse_symbol,
                start=start_date,
                end=end_date
            )
            
            if not data.empty:
                # Rename columns to match yfinance format
                data = data.rename(columns={
                    'Open': 'Open',
                    'High': 'High', 
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                })
                return data
        except Exception as e:
            print(f"NSE error: {e}")
        return None
    
    def get_multi_timeframe_data(self):
        """Fetch data for all required timeframes"""
        timeframes = {
            '5m': '5m',
            '15m': '15m', 
            '1h': '1h',
            '1d': '1d',
            '1wk': '1wk'
        }
        
        data_dict = {}
        
        for tf_name, tf_code in timeframes.items():
            print(f"Fetching {tf_name} data...")
            
            # Try different periods for different timeframes
            if tf_code in ['5m', '15m']:
                period = "60d"  # 60 days for intraday
            elif tf_code == '1h':
                period = "730d"  # 2 years for hourly
            else:
                period = "max"  # Maximum for daily/weekly
            
            data = self.fetch_yfinance_data(period=period, interval=tf_code)
            
            if data is not None and not data.empty:
                data_dict[tf_name] = data
                print(f"✓ {tf_name}: {len(data)} records from {data.index[0]} to {data.index[-1]}")
            else:
                print(f"✗ Failed to fetch {tf_name} data")
        
        # Also try NSE data for daily timeframe as backup
        nse_data = self.fetch_nse_data()
        if nse_data is not None and not nse_data.empty:
            data_dict['1d_nse'] = nse_data
            print(f"✓ NSE Daily: {len(nse_data)} records")
        
        return data_dict
    
    def get_live_data(self, interval="1m"):
        """Get live/latest data"""
        try:
            ticker = yf.Ticker(self.symbol)
            # Get last few periods for live data
            data = ticker.history(period="1d", interval=interval)
            return data.tail(1) if not data.empty else None
        except Exception as e:
            print(f"Live data error: {e}")
            return None
    
    def validate_and_clean_data(self, data):
        """Clean and validate data"""
        if data is None or data.empty:
            return None
        
        # Remove rows with missing OHLC data
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Ensure High >= Low
        invalid_rows = data['High'] < data['Low']
        if invalid_rows.any():
            print(f"Removing {invalid_rows.sum()} invalid rows where High < Low")
            data = data[~invalid_rows]
        
        # Ensure High >= Open, Close and Low <= Open, Close
        data['High'] = np.maximum.reduce([data['High'], data['Open'], data['Close']])
        data['Low'] = np.minimum.reduce([data['Low'], data['Open'], data['Close']])
        
        # Fill missing volume with 0
        data['Volume'] = data['Volume'].fillna(0)
        
        return data
    
    def get_comprehensive_data(self):
        """Get comprehensive data for all timeframes"""
        print(f"Fetching comprehensive data for {self.symbol}...")
        
        raw_data = self.get_multi_timeframe_data()
        cleaned_data = {}
        
        for timeframe, data in raw_data.items():
            cleaned = self.validate_and_clean_data(data)
            if cleaned is not None and not cleaned.empty:
                cleaned_data[timeframe] = cleaned
        
        return cleaned_data
    
    def get_stock_info(self):
        """Get additional stock information"""
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            return {
                'company_name': info.get('longName', self.symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
        except Exception as e:
            print(f"Stock info error: {e}")
            return {'company_name': self.symbol}

# Test the data fetcher
if __name__ == "__main__":
    fetcher = IndianStockDataFetcher("RELIANCE.NS")
    data = fetcher.get_comprehensive_data()
    info = fetcher.get_stock_info()
    
    print(f"\nStock Info: {info}")
    print(f"\nAvailable timeframes: {list(data.keys())}")
    for tf, df in data.items():
        print(f"{tf}: {len(df)} records")