#!/usr/bin/env python3
"""
COMPLETE MULTI-TIMEFRAME AUTONOMOUS DEEP LEARNING ANALYSIS SYSTEM
================================================================

This system provides:
1. Multi-timeframe analysis (5m, 15m, 1h, 1d, 1w)
2. Separate Technical and Price Action reports
3. Professional candlestick charts with pattern overlays
4. Detailed ML predictions with confidence intervals
5. Advanced pattern evolution tracking and success rates
6. Time-based cyclical analysis
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import yfinance as yf
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import zscore

# ML Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteMultiTimeframeAnalyzer:
    """
    Complete Multi-Timeframe Autonomous Deep Learning Analysis System
    """
    
    def __init__(self, symbol="^NSEI"):
        self.symbol = symbol
        self.company_name = self._get_company_name(symbol)
        self.output_dir = f"{symbol.replace('^', '').replace('.', '_').lower()}_complete_analysis"
        self.ensure_output_directory()
        
        # Data storage for multiple timeframes
        self.timeframe_data = {}
        self.timeframe_patterns = {}
        self.timeframe_indicators = {}
        self.timeframe_predictions = {}
        
        # Analysis results
        self.technical_analysis_report = {}
        self.price_action_report = {}
        self.pattern_evolution = {}
        self.success_rates = {}
        
        print(f"🧠 Initializing Complete Multi-Timeframe Analyzer for {self.company_name}")
        
    def _get_company_name(self, symbol):
        """Get company name based on symbol"""
        names = {
            "^NSEI": "NIFTY 50 Index",
            "RELIANCE.NS": "Reliance Industries Limited",
            "TCS.NS": "Tata Consultancy Services"
        }
        return names.get(symbol, symbol)
        
    def ensure_output_directory(self):
        """Create output directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/charts", exist_ok=True)
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
        
    def fetch_multi_timeframe_data(self):
        """Fetch comprehensive multi-timeframe data"""
        print("📊 Fetching Multi-Timeframe Data...")
        
        timeframes = {
            '5m': {'period': '5d', 'interval': '5m', 'name': '5-Minute'},
            '15m': {'period': '1mo', 'interval': '15m', 'name': '15-Minute'},
            '1h': {'period': '3mo', 'interval': '1h', 'name': '1-Hour'},
            '1d': {'period': '2y', 'interval': '1d', 'name': 'Daily'},
            '1wk': {'period': '5y', 'interval': '1wk', 'name': 'Weekly'}
        }
        
        ticker = yf.Ticker(self.symbol)
        
        for tf_key, tf_config in timeframes.items():
            try:
                print(f"  Fetching {tf_config['name']} data...")
                df = ticker.history(period=tf_config['period'], interval=tf_config['interval'])
                
                if not df.empty:
                    # Clean timezone info
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    self.timeframe_data[tf_key] = {
                        'data': df,
                        'name': tf_config['name'],
                        'interval': tf_config['interval']
                    }
                    print(f"    ✓ {tf_config['name']}: {len(df)} records")
                else:
                    print(f"    ✗ No data for {tf_config['name']}")
                    
            except Exception as e:
                print(f"    ✗ Error fetching {tf_config['name']}: {e}")
                
        try:
            self.info = ticker.info
        except:
            self.info = {"longName": self.company_name}
            
    def analyze_all_timeframes(self):
        """Analyze all timeframes comprehensively"""
        print("🔍 Analyzing All Timeframes...")
        
        for tf_key, tf_data in self.timeframe_data.items():
            print(f"  Analyzing {tf_data['name']}...")
            df = tf_data['data']
            
            # Pattern Discovery
            patterns = self.discover_patterns_advanced(df, tf_key)
            self.timeframe_patterns[tf_key] = patterns
            
            # Technical Analysis
            indicators = self.calculate_advanced_indicators(df, tf_key)
            self.timeframe_indicators[tf_key] = indicators
            
            # ML Predictions
            predictions = self.generate_ml_predictions(df, tf_key)
            self.timeframe_predictions[tf_key] = predictions
            
            # Pattern Evolution
            evolution = self.track_pattern_evolution(df, patterns, tf_key)
            self.pattern_evolution[tf_key] = evolution
            
            print(f"    ✓ {tf_data['name']}: {len(patterns)} patterns, {len(indicators)} indicators")
            
    def discover_patterns_advanced(self, df, timeframe):
        """Advanced pattern discovery with success rate tracking"""
        patterns = {}
        
        # Traditional Candlestick Patterns
        patterns.update(self._detect_candlestick_patterns_manual(df))
        
        # Chart Patterns
        patterns.update(self._detect_chart_patterns_advanced(df))
        
        # ML-Discovered Patterns
        patterns.update(self._discover_ml_patterns_advanced(df))
        
        # Volume Patterns
        patterns.update(self._detect_volume_patterns_advanced(df))
        
        # Calculate success rates for each pattern
        for pattern_name, pattern_series in patterns.items():
            if isinstance(pattern_series, pd.Series) and pattern_series.any():
                success_rate = self._calculate_pattern_success_rate(df, pattern_series, timeframe)
                patterns[f"{pattern_name}_success_rate"] = success_rate
        
        return patterns
        
    def _detect_candlestick_patterns_manual(self, df):
        """Manual candlestick pattern detection"""
        patterns = {}
        
        open_price = df['Open'].values
        high_price = df['High'].values
        low_price = df['Low'].values
        close_price = df['Close'].values
        
        # Calculate pattern components
        body = close_price - open_price
        upper_shadow = high_price - np.maximum(open_price, close_price)
        lower_shadow = np.minimum(open_price, close_price) - low_price
        total_range = high_price - low_price
        body_size = np.abs(body)
        
        # Single candlestick patterns with detailed criteria
        patterns['Doji'] = (body_size <= total_range * 0.1) & (total_range > 0)
        patterns['Hammer'] = (lower_shadow >= body_size * 2) & (upper_shadow <= body_size * 0.5) & (body < 0)
        patterns['Shooting_Star'] = (upper_shadow >= body_size * 2) & (lower_shadow <= body_size * 0.5) & (body > 0)
        patterns['Hanging_Man'] = (lower_shadow >= body_size * 2) & (upper_shadow <= body_size * 0.5) & (body > 0)
        patterns['Inverted_Hammer'] = (upper_shadow >= body_size * 2) & (lower_shadow <= body_size * 0.5) & (body < 0)
        patterns['Marubozu'] = (upper_shadow <= total_range * 0.05) & (lower_shadow <= total_range * 0.05) & (body_size >= total_range * 0.9)
        patterns['Spinning_Top'] = (body_size <= total_range * 0.3) & (upper_shadow >= body_size * 0.5) & (lower_shadow >= body_size * 0.5)
        patterns['Long_Legged_Doji'] = (body_size <= total_range * 0.05) & (upper_shadow >= total_range * 0.3) & (lower_shadow >= total_range * 0.3)
        patterns['Dragonfly_Doji'] = (body_size <= total_range * 0.1) & (upper_shadow <= total_range * 0.1) & (lower_shadow >= total_range * 0.6)
        patterns['Gravestone_Doji'] = (body_size <= total_range * 0.1) & (lower_shadow <= total_range * 0.1) & (upper_shadow >= total_range * 0.6)
        
        # Two candlestick patterns
        prev_body = np.roll(body, 1)
        prev_high = np.roll(high_price, 1)
        prev_low = np.roll(low_price, 1)
        prev_open = np.roll(open_price, 1)
        prev_close = np.roll(close_price, 1)
        prev_body_size = np.roll(body_size, 1)
        
        patterns['Engulfing_Bull'] = (body > 0) & (prev_body < 0) & (open_price < prev_close) & (close_price > prev_open) & (body_size > prev_body_size * 1.1)
        patterns['Engulfing_Bear'] = (body < 0) & (prev_body > 0) & (open_price > prev_close) & (close_price < prev_open) & (body_size > prev_body_size * 1.1)
        patterns['Harami_Bull'] = (body > 0) & (prev_body < 0) & (open_price > prev_close) & (close_price < prev_open) & (body_size < prev_body_size * 0.8)
        patterns['Harami_Bear'] = (body < 0) & (prev_body > 0) & (open_price < prev_close) & (close_price > prev_open) & (body_size < prev_body_size * 0.8)
        patterns['Piercing'] = (body > 0) & (prev_body < 0) & (open_price < prev_low) & (close_price > (prev_open + prev_close) / 2) & (close_price < prev_open)
        patterns['Dark_Cloud'] = (body < 0) & (prev_body > 0) & (open_price > prev_high) & (close_price < (prev_open + prev_close) / 2) & (close_price > prev_open)
        patterns['Tweezer_Tops'] = (np.abs(high_price - prev_high) / high_price < 0.001) & (body < 0) & (prev_body > 0)
        patterns['Tweezer_Bottoms'] = (np.abs(low_price - prev_low) / low_price < 0.001) & (body > 0) & (prev_body < 0)
        
        # Three candlestick patterns
        patterns['Morning_Star'] = self._detect_three_candle_pattern(df, 'morning_star')
        patterns['Evening_Star'] = self._detect_three_candle_pattern(df, 'evening_star')
        patterns['Three_White_Soldiers'] = (body > 0) & (prev_body > 0) & np.roll((body > 0), 2) & (close_price > prev_close) & (prev_close > np.roll(close_price, 2))
        patterns['Three_Black_Crows'] = (body < 0) & (prev_body < 0) & np.roll((body < 0), 2) & (close_price < prev_close) & (prev_close < np.roll(close_price, 2))
        patterns['Three_Inside_Up'] = patterns['Harami_Bull'] & np.roll((body > 0) & (close_price > np.roll(close_price, 2)), -1)
        patterns['Three_Inside_Down'] = patterns['Harami_Bear'] & np.roll((body < 0) & (close_price < np.roll(close_price, 2)), -1)
        
        return {k: pd.Series(v, index=df.index) for k, v in patterns.items()}
        
    def _detect_three_candle_pattern(self, df, pattern_type):
        """Detect three-candle patterns"""
        close_price = df['Close'].values
        open_price = df['Open'].values
        high_price = df['High'].values
        low_price = df['Low'].values
        
        body = close_price - open_price
        body_size = np.abs(body)
        total_range = high_price - low_price
        
        if pattern_type == 'morning_star':
            # Day 1: Large bearish candle
            day1_bear = np.roll(body < -total_range * 0.05, 2)
            # Day 2: Small body (star)
            day2_small = np.roll(body_size < total_range * 0.3, 1)
            # Day 3: Large bullish candle
            day3_bull = body > total_range * 0.05
            # Gap conditions
            gap_down = np.roll(high_price, 1) < np.roll(low_price, 2)
            gap_up = low_price > np.roll(high_price, 1)
            
            return day1_bear & day2_small & day3_bull & gap_down & gap_up
            
        elif pattern_type == 'evening_star':
            # Day 1: Large bullish candle
            day1_bull = np.roll(body > total_range * 0.05, 2)
            # Day 2: Small body (star)
            day2_small = np.roll(body_size < total_range * 0.3, 1)
            # Day 3: Large bearish candle
            day3_bear = body < -total_range * 0.05
            # Gap conditions
            gap_up = np.roll(low_price, 1) > np.roll(high_price, 2)
            gap_down = high_price < np.roll(low_price, 1)
            
            return day1_bull & day2_small & day3_bear & gap_up & gap_down
            
        return np.zeros(len(df), dtype=bool)
        
    def _detect_chart_patterns_advanced(self, df):
        """Advanced chart pattern detection"""
        patterns = {}
        
        # Support and Resistance levels
        highs = df['High'].rolling(20).max()
        lows = df['Low'].rolling(20).min()
        
        # Triangle patterns with volume confirmation
        patterns['Ascending_Triangle'] = self._detect_ascending_triangle_advanced(df)
        patterns['Descending_Triangle'] = self._detect_descending_triangle_advanced(df)
        patterns['Symmetrical_Triangle'] = self._detect_symmetrical_triangle_advanced(df)
        
        # Head and Shoulders patterns
        patterns['Head_Shoulders'] = self._detect_head_shoulders_advanced(df)
        patterns['Inverse_Head_Shoulders'] = self._detect_inverse_head_shoulders_advanced(df)
        
        # Flag and Pennant patterns
        patterns['Bull_Flag'] = self._detect_bull_flag_advanced(df)
        patterns['Bear_Flag'] = self._detect_bear_flag_advanced(df)
        patterns['Pennant'] = self._detect_pennant_advanced(df)
        
        # Channel patterns
        patterns['Rising_Channel'] = self._detect_rising_channel_advanced(df)
        patterns['Falling_Channel'] = self._detect_falling_channel_advanced(df)
        patterns['Horizontal_Channel'] = self._detect_horizontal_channel_advanced(df)
        
        # Wedge patterns
        patterns['Rising_Wedge'] = self._detect_rising_wedge_advanced(df)
        patterns['Falling_Wedge'] = self._detect_falling_wedge_advanced(df)
        
        # Double patterns
        patterns['Double_Top'] = self._detect_double_top_advanced(df)
        patterns['Double_Bottom'] = self._detect_double_bottom_advanced(df)
        
        # Cup and Handle
        patterns['Cup_Handle'] = self._detect_cup_handle_advanced(df)
        
        return patterns
        
    def _detect_ascending_triangle_advanced(self, df):
        """Advanced ascending triangle detection with volume confirmation"""
        highs = df['High'].rolling(10).max()
        lows = df['Low'].rolling(10).min()
        volume = df['Volume']
        
        # Horizontal resistance
        resistance_level = highs.rolling(30).max()
        resistance_touches = (df['High'] >= resistance_level * 0.998).rolling(50).sum()
        resistance_horizontal = resistance_level.rolling(20).std() / resistance_level.rolling(20).mean() < 0.02
        
        # Rising support
        support_slope = lows.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 10 else 0)
        support_rising = support_slope > 0
        
        # Volume pattern (decreasing during formation)
        volume_declining = volume.rolling(20).mean() < volume.rolling(50).mean()
        
        # Convergence
        range_narrowing = (highs - lows).rolling(20).apply(lambda x: x.iloc[-1] < x.iloc[0] * 0.9 if len(x) >= 20 else False)
        
        return resistance_horizontal & support_rising & resistance_touches >= 2 & range_narrowing
        
    def _detect_descending_triangle_advanced(self, df):
        """Advanced descending triangle detection"""
        highs = df['High'].rolling(10).max()
        lows = df['Low'].rolling(10).min()
        
        # Horizontal support
        support_level = lows.rolling(30).min()
        support_touches = (df['Low'] <= support_level * 1.002).rolling(50).sum()
        support_horizontal = support_level.rolling(20).std() / support_level.rolling(20).mean() < 0.02
        
        # Falling resistance
        resistance_slope = highs.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 10 else 0)
        resistance_falling = resistance_slope < 0
        
        # Convergence
        range_narrowing = (highs - lows).rolling(20).apply(lambda x: x.iloc[-1] < x.iloc[0] * 0.9 if len(x) >= 20 else False)
        
        return support_horizontal & resistance_falling & support_touches >= 2 & range_narrowing
        
    def _detect_symmetrical_triangle_advanced(self, df):
        """Advanced symmetrical triangle detection"""
        highs = df['High'].rolling(10).max()
        lows = df['Low'].rolling(10).min()
        
        # Converging trend lines
        resistance_slope = highs.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 10 else 0)
        support_slope = lows.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 10 else 0)
        
        resistance_falling = resistance_slope < -0.001
        support_rising = support_slope > 0.001
        
        # Convergence rate
        convergence = np.abs(resistance_slope + support_slope) < np.abs(resistance_slope - support_slope) * 0.5
        
        # Volume pattern
        volume_declining = df['Volume'].rolling(20).mean() < df['Volume'].rolling(50).mean()
        
        return resistance_falling & support_rising & convergence & volume_declining
        
    def _detect_head_shoulders_advanced(self, df):
        """Advanced head and shoulders pattern detection"""
        highs = df['High'].rolling(5).max()
        lows = df['Low'].rolling(5).min()
        
        # Find peaks
        try:
            peaks, _ = find_peaks(highs, distance=20, prominence=highs.std()*0.5)
            
            if len(peaks) >= 3:
                pattern = pd.Series(False, index=df.index)
                
                for i in range(len(peaks)-2):
                    left_shoulder = highs.iloc[peaks[i]]
                    head = highs.iloc[peaks[i+1]]
                    right_shoulder = highs.iloc[peaks[i+2]]
                    
                    # Head higher than shoulders
                    head_prominence = (head > left_shoulder * 1.02) and (head > right_shoulder * 1.02)
                    
                    # Shoulders approximately equal
                    shoulder_symmetry = abs(left_shoulder - right_shoulder) / left_shoulder < 0.05
                    
                    # Neckline analysis
                    neckline_start = lows.iloc[peaks[i]:peaks[i+1]].min()
                    neckline_end = lows.iloc[peaks[i+1]:peaks[i+2]].min()
                    neckline_level = min(neckline_start, neckline_end)
                    
                    # Volume confirmation (decreasing on right shoulder)
                    volume_confirmation = df['Volume'].iloc[peaks[i+2]] < df['Volume'].iloc[peaks[i]]
                    
                    if head_prominence and shoulder_symmetry and volume_confirmation:
                        pattern.iloc[peaks[i]:peaks[i+2]+10] = True
                        
                return pattern
        except:
            pass
            
        return pd.Series(False, index=df.index)
        
    def _detect_inverse_head_shoulders_advanced(self, df):
        """Advanced inverse head and shoulders pattern detection"""
        lows = df['Low'].rolling(5).min()
        highs = df['High'].rolling(5).max()
        
        # Find troughs (inverse of peaks)
        try:
            troughs, _ = find_peaks(-lows, distance=20, prominence=lows.std()*0.5)
            
            if len(troughs) >= 3:
                pattern = pd.Series(False, index=df.index)
                
                for i in range(len(troughs)-2):
                    left_shoulder = lows.iloc[troughs[i]]
                    head = lows.iloc[troughs[i+1]]
                    right_shoulder = lows.iloc[troughs[i+2]]
                    
                    # Head lower than shoulders
                    head_prominence = (head < left_shoulder * 0.98) and (head < right_shoulder * 0.98)
                    
                    # Shoulders approximately equal
                    shoulder_symmetry = abs(left_shoulder - right_shoulder) / left_shoulder < 0.05
                    
                    # Volume confirmation (increasing on right shoulder)
                    volume_confirmation = df['Volume'].iloc[troughs[i+2]] > df['Volume'].iloc[troughs[i]]
                    
                    if head_prominence and shoulder_symmetry and volume_confirmation:
                        pattern.iloc[troughs[i]:troughs[i+2]+10] = True
                        
                return pattern
        except:
            pass
            
        return pd.Series(False, index=df.index)
        
    def _detect_bull_flag_advanced(self, df):
        """Advanced bull flag pattern detection"""
        returns = df['Close'].pct_change()
        volume = df['Volume']
        volume_avg = volume.rolling(20).mean()
        
        # Strong upward move (flagpole)
        flagpole_strength = returns.rolling(10).sum() > 0.08
        flagpole_volume = volume.rolling(10).mean() > volume_avg * 1.2
        
        # Consolidation (flag)
        consolidation_range = (df['High'].rolling(15).max() - df['Low'].rolling(15).min()) / df['Close'] < 0.05
        consolidation_slope = df['Close'].rolling(15).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 15 else 0)
        slight_decline = (consolidation_slope < 0) & (consolidation_slope > -0.002)
        
        # Volume decline during consolidation
        volume_decline = volume < volume_avg * 0.8
        
        # Time factor (flag should be shorter than pole)
        time_factor = True  # Simplified for this implementation
        
        return flagpole_strength.shift(15) & flagpole_volume.shift(15) & consolidation_range & slight_decline & volume_decline
        
    def _detect_bear_flag_advanced(self, df):
        """Advanced bear flag pattern detection"""
        returns = df['Close'].pct_change()
        volume = df['Volume']
        volume_avg = volume.rolling(20).mean()
        
        # Strong downward move (flagpole)
        flagpole_strength = returns.rolling(10).sum() < -0.08
        flagpole_volume = volume.rolling(10).mean() > volume_avg * 1.2
        
        # Consolidation (flag)
        consolidation_range = (df['High'].rolling(15).max() - df['Low'].rolling(15).min()) / df['Close'] < 0.05
        consolidation_slope = df['Close'].rolling(15).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 15 else 0)
        slight_incline = (consolidation_slope > 0) & (consolidation_slope < 0.002)
        
        # Volume decline during consolidation
        volume_decline = volume < volume_avg * 0.8
        
        return flagpole_strength.shift(15) & flagpole_volume.shift(15) & consolidation_range & slight_incline & volume_decline
        
    def _detect_pennant_advanced(self, df):
        """Advanced pennant pattern detection"""
        highs = df['High'].rolling(3).max()
        lows = df['Low'].rolling(3).min()
        volume = df['Volume']
        
        # Converging price action after strong move
        range_start = (highs - lows).rolling(20).apply(lambda x: x.iloc[0] if len(x) >= 20 else 0)
        range_current = highs - lows
        range_contracting = range_current < range_start * 0.6
        
        # Symmetrical convergence
        upper_slope = highs.rolling(15).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 15 else 0)
        lower_slope = lows.rolling(15).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 15 else 0)
        convergence = (upper_slope < 0) & (lower_slope > 0) & (np.abs(upper_slope + lower_slope) < 0.001)
        
        # Volume decline
        volume_decline = volume < volume.rolling(20).mean() * 0.7
        
        return range_contracting & convergence & volume_decline
        
    def _detect_rising_channel_advanced(self, df):
        """Advanced rising channel detection"""
        highs = df['High'].rolling(5).max()
        lows = df['Low'].rolling(5).min()
        
        # Parallel rising trend lines
        upper_slope = highs.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 30 else 0)
        lower_slope = lows.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 30 else 0)
        
        # Both slopes positive and similar
        both_rising = (upper_slope > 0.001) & (lower_slope > 0.001)
        parallel = np.abs(upper_slope - lower_slope) / np.maximum(upper_slope, lower_slope) < 0.3
        
        # Multiple touches of both lines
        upper_touches = (df['High'] >= highs * 0.998).rolling(50).sum()
        lower_touches = (df['Low'] <= lows * 1.002).rolling(50).sum()
        sufficient_touches = (upper_touches >= 2) & (lower_touches >= 2)
        
        return both_rising & parallel & sufficient_touches
        
    def _detect_falling_channel_advanced(self, df):
        """Advanced falling channel detection"""
        highs = df['High'].rolling(5).max()
        lows = df['Low'].rolling(5).min()
        
        # Parallel falling trend lines
        upper_slope = highs.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 30 else 0)
        lower_slope = lows.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 30 else 0)
        
        # Both slopes negative and similar
        both_falling = (upper_slope < -0.001) & (lower_slope < -0.001)
        parallel = np.abs(upper_slope - lower_slope) / np.maximum(np.abs(upper_slope), np.abs(lower_slope)) < 0.3
        
        # Multiple touches of both lines
        upper_touches = (df['High'] >= highs * 0.998).rolling(50).sum()
        lower_touches = (df['Low'] <= lows * 1.002).rolling(50).sum()
        sufficient_touches = (upper_touches >= 2) & (lower_touches >= 2)
        
        return both_falling & parallel & sufficient_touches
        
    def _detect_horizontal_channel_advanced(self, df):
        """Advanced horizontal channel detection"""
        highs = df['High'].rolling(10).max()
        lows = df['Low'].rolling(10).min()
        
        # Horizontal resistance and support
        resistance_level = highs.rolling(30).max()
        support_level = lows.rolling(30).min()
        
        # Check for horizontal levels (low slope)
        resistance_slope = resistance_level.rolling(20).apply(lambda x: abs(stats.linregress(range(len(x)), x)[0]) if len(x) >= 20 else 1)
        support_slope = support_level.rolling(20).apply(lambda x: abs(stats.linregress(range(len(x)), x)[0]) if len(x) >= 20 else 1)
        
        horizontal_resistance = resistance_slope < 0.001
        horizontal_support = support_slope < 0.001
        
        # Multiple touches
        resistance_touches = (df['High'] >= resistance_level * 0.998).rolling(50).sum()
        support_touches = (df['Low'] <= support_level * 1.002).rolling(50).sum()
        
        # Channel width consistency
        channel_width = (resistance_level - support_level) / df['Close']
        width_consistent = channel_width.rolling(20).std() / channel_width.rolling(20).mean() < 0.2
        
        return horizontal_resistance & horizontal_support & (resistance_touches >= 2) & (support_touches >= 2) & width_consistent
        
    def _detect_rising_wedge_advanced(self, df):
        """Advanced rising wedge detection"""
        highs = df['High'].rolling(5).max()
        lows = df['Low'].rolling(5).min()
        volume = df['Volume']
        
        # Both trend lines rising but converging
        upper_slope = highs.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 30 else 0)
        lower_slope = lows.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 30 else 0)
        
        both_rising = (upper_slope > 0) & (lower_slope > 0)
        converging = lower_slope > upper_slope  # Lower line rising faster
        
        # Range contraction
        range_contracting = (highs - lows).rolling(20).apply(lambda x: x.iloc[-1] < x.iloc[0] * 0.8 if len(x) >= 20 else False)
        
        # Volume decline (bearish divergence)
        volume_decline = volume.rolling(20).mean() < volume.rolling(50).mean()
        
        return both_rising & converging & range_contracting & volume_decline
        
    def _detect_falling_wedge_advanced(self, df):
        """Advanced falling wedge detection"""
        highs = df['High'].rolling(5).max()
        lows = df['Low'].rolling(5).min()
        volume = df['Volume']
        
        # Both trend lines falling but converging
        upper_slope = highs.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 30 else 0)
        lower_slope = lows.rolling(30).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 30 else 0)
        
        both_falling = (upper_slope < 0) & (lower_slope < 0)
        converging = upper_slope > lower_slope  # Upper line falling faster
        
        # Range contraction
        range_contracting = (highs - lows).rolling(20).apply(lambda x: x.iloc[-1] < x.iloc[0] * 0.8 if len(x) >= 20 else False)
        
        # Volume decline then increase (bullish divergence)
        volume_pattern = volume.rolling(10).mean() > volume.rolling(30).mean()
        
        return both_falling & converging & range_contracting & volume_pattern
        
    def _detect_double_top_advanced(self, df):
        """Advanced double top detection"""
        highs = df['High'].rolling(5).max()
        
        try:
            peaks, _ = find_peaks(highs, distance=30, prominence=highs.std()*0.5)
            
            if len(peaks) >= 2:
                pattern = pd.Series(False, index=df.index)
                
                for i in range(len(peaks)-1):
                    peak1 = highs.iloc[peaks[i]]
                    peak2 = highs.iloc[peaks[i+1]]
                    
                    # Peaks approximately equal
                    peak_similarity = abs(peak1 - peak2) / peak1 < 0.03
                    
                    # Valley between peaks
                    valley_start = peaks[i] + 5
                    valley_end = peaks[i+1] - 5
                    if valley_end > valley_start:
                        valley_low = df['Low'].iloc[valley_start:valley_end].min()
                        valley_depth = (min(peak1, peak2) - valley_low) / min(peak1, peak2) > 0.05
                        
                        # Volume confirmation (second peak lower volume)
                        volume_confirmation = df['Volume'].iloc[peaks[i+1]] < df['Volume'].iloc[peaks[i]]
                        
                        if peak_similarity and valley_depth and volume_confirmation:
                            pattern.iloc[peaks[i]:peaks[i+1]+10] = True
                            
                return pattern
        except:
            pass
            
        return pd.Series(False, index=df.index)
        
    def _detect_double_bottom_advanced(self, df):
        """Advanced double bottom detection"""
        lows = df['Low'].rolling(5).min()
        
        try:
            troughs, _ = find_peaks(-lows, distance=30, prominence=lows.std()*0.5)
            
            if len(troughs) >= 2:
                pattern = pd.Series(False, index=df.index)
                
                for i in range(len(troughs)-1):
                    trough1 = lows.iloc[troughs[i]]
                    trough2 = lows.iloc[troughs[i+1]]
                    
                    # Troughs approximately equal
                    trough_similarity = abs(trough1 - trough2) / trough1 < 0.03
                    
                    # Peak between troughs
                    peak_start = troughs[i] + 5
                    peak_end = troughs[i+1] - 5
                    if peak_end > peak_start:
                        peak_high = df['High'].iloc[peak_start:peak_end].max()
                        peak_height = (peak_high - max(trough1, trough2)) / max(trough1, trough2) > 0.05
                        
                        # Volume confirmation (second trough higher volume)
                        volume_confirmation = df['Volume'].iloc[troughs[i+1]] > df['Volume'].iloc[troughs[i]]
                        
                        if trough_similarity and peak_height and volume_confirmation:
                            pattern.iloc[troughs[i]:troughs[i+1]+10] = True
                            
                return pattern
        except:
            pass
            
        return pd.Series(False, index=df.index)
        
    def _detect_cup_handle_advanced(self, df):
        """Advanced cup and handle detection"""
        highs = df['High'].rolling(5).max()
        lows = df['Low'].rolling(5).min()
        volume = df['Volume']
        
        # Cup formation (U-shaped)
        cup_length = 50  # Minimum 50 periods for cup
        
        pattern = pd.Series(False, index=df.index)
        
        for i in range(cup_length, len(df)-20):
            cup_start = i - cup_length
            cup_mid = i - cup_length//2
            cup_end = i
            
            # Left rim, bottom, right rim
            left_rim = highs.iloc[cup_start:cup_start+5].max()
            bottom = lows.iloc[cup_mid-10:cup_mid+10].min()
            right_rim = highs.iloc[cup_end-5:cup_end].max()
            
            # Cup criteria
            rim_similarity = abs(left_rim - right_rim) / left_rim < 0.05
            cup_depth = (left_rim - bottom) / left_rim > 0.15
            cup_shape = bottom < left_rim * 0.9 and bottom < right_rim * 0.9
            
            # Handle formation (small pullback after right rim)
            handle_start = cup_end
            handle_end = min(handle_start + 15, len(df))
            if handle_end > handle_start:
                handle_low = lows.iloc[handle_start:handle_end].min()
                handle_pullback = (right_rim - handle_low) / right_rim < 0.15
                handle_volume_decline = volume.iloc[handle_start:handle_end].mean() < volume.iloc[cup_start:cup_end].mean()
                
                if rim_similarity and cup_depth and cup_shape and handle_pullback and handle_volume_decline:
                    pattern.iloc[handle_start:handle_end] = True
                    
        return pattern
        
    def _discover_ml_patterns_advanced(self, df):
        """Advanced ML pattern discovery with clustering and anomaly detection"""
        patterns = {}
        
        try:
            # Create comprehensive feature matrix
            features = self._create_advanced_pattern_features(df)
            
            if len(features) > 100:
                # KMeans clustering for pattern discovery
                kmeans = KMeans(n_clusters=12, random_state=42)
                clusters = kmeans.fit_predict(features)
                
                # Create cluster-based patterns
                for i in range(12):
                    cluster_mask = clusters == i
                    if cluster_mask.sum() > 5:  # Only keep clusters with sufficient points
                        patterns[f'ML_Cluster_Pattern_{i+1}'] = pd.Series(cluster_mask, index=df.index)
                
                # DBSCAN for density-based patterns
                dbscan = DBSCAN(eps=0.8, min_samples=8)
                density_clusters = dbscan.fit_predict(features)
                
                unique_clusters = np.unique(density_clusters)
                for cluster in unique_clusters:
                    if cluster != -1:  # Exclude noise
                        cluster_mask = density_clusters == cluster
                        if cluster_mask.sum() > 3:
                            patterns[f'ML_Density_Pattern_{cluster+1}'] = pd.Series(cluster_mask, index=df.index)
                
                # Isolation Forest for anomaly detection
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso_forest.fit_predict(features)
                patterns['ML_Anomaly_Pattern'] = pd.Series(anomalies == -1, index=df.index)
                
                # Statistical anomalies
                patterns['ML_Statistical_Outlier'] = self._detect_statistical_outliers(df)
                patterns['ML_Volume_Anomaly'] = self._detect_volume_anomalies(df)
                patterns['ML_Price_Gap_Pattern'] = self._detect_price_gaps(df)
                
        except Exception as e:
            print(f"ML pattern discovery error: {e}")
            # Fallback patterns
            returns = df['Close'].pct_change()
            patterns['ML_High_Volatility'] = np.abs(returns) > returns.std() * 2.5
            patterns['ML_Momentum_Shift'] = returns.rolling(5).mean().diff().abs() > 0.015
            
        return patterns
        
    def _create_advanced_pattern_features(self, df):
        """Create comprehensive feature matrix for ML pattern discovery"""
        features_df = pd.DataFrame(index=df.index)
        
        # Price features
        returns = df['Close'].pct_change()
        features_df['return'] = returns
        features_df['log_return'] = np.log(df['Close'] / df['Close'].shift())
        features_df['return_volatility'] = returns.rolling(10).std()
        
        # Candlestick features
        body = df['Close'] - df['Open']
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        total_range = df['High'] - df['Low']
        
        features_df['body_size'] = np.abs(body) / df['Close']
        features_df['upper_shadow_ratio'] = upper_shadow / total_range
        features_df['lower_shadow_ratio'] = lower_shadow / total_range
        features_df['body_position'] = (df['Close'] - df['Low']) / total_range
        
        # Volume features
        features_df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features_df['volume_price_correlation'] = returns.rolling(20).corr(df['Volume'].pct_change())
        features_df['volume_volatility'] = df['Volume'].rolling(10).std() / df['Volume'].rolling(10).mean()
        
        # Momentum features
        features_df['roc_5'] = df['Close'].pct_change(5)
        features_df['roc_10'] = df['Close'].pct_change(10)
        features_df['momentum'] = returns.rolling(10).mean()
        features_df['acceleration'] = returns.rolling(5).mean().diff()
        
        # Technical indicator features
        features_df['rsi'] = self._calculate_rsi_simple(df['Close'])
        features_df['macd'] = self._calculate_macd_simple(df['Close'])
        features_df['bb_position'] = self._calculate_bb_position_simple(df['Close'])
        
        # Price action features
        features_df['range_expansion'] = total_range / total_range.rolling(20).mean()
        features_df['gap_up'] = (df['Open'] > df['High'].shift()) * 1.0
        features_df['gap_down'] = (df['Open'] < df['Low'].shift()) * 1.0
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'return_lag_{lag}'] = returns.shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['volume_ratio'].shift(lag)
            
        return features_df.fillna(0).values
        
    def _calculate_rsi_simple(self, prices, period=14):
        """Simple RSI calculation"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd_simple(self, prices):
        """Simple MACD calculation"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        return ema_12 - ema_26
        
    def _calculate_bb_position_simple(self, prices, period=20):
        """Simple Bollinger Band position"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return (prices - lower) / (upper - lower)
        
    def _detect_statistical_outliers(self, df):
        """Detect statistical outliers in price movements"""
        returns = df['Close'].pct_change()
        z_scores = np.abs(zscore(returns.dropna()))
        outliers = pd.Series(False, index=df.index)
        outliers.iloc[1:] = z_scores > 3  # 3-sigma outliers
        return outliers
        
    def _detect_volume_anomalies(self, df):
        """Detect volume anomalies"""
        volume_z = np.abs(zscore(df['Volume']))
        return volume_z > 2.5
        
    def _detect_price_gaps(self, df):
        """Detect significant price gaps"""
        gap_up = (df['Open'] > df['High'].shift() * 1.005)
        gap_down = (df['Open'] < df['Low'].shift() * 0.995)
        return gap_up | gap_down
        
    def _detect_volume_patterns_advanced(self, df):
        """Advanced volume pattern detection"""
        patterns = {}
        
        volume = df['Volume']
        price_change = df['Close'].pct_change()
        volume_sma = volume.rolling(20).mean()
        volume_std = volume.rolling(20).std()
        
        # Basic volume patterns
        patterns['Volume_Spike'] = volume > volume_sma + 2 * volume_std
        patterns['Volume_Dry_Up'] = volume < volume_sma - volume_std
        patterns['Volume_Climax'] = (volume > volume_sma * 3) & (np.abs(price_change) > price_change.std() * 2)
        
        # Advanced volume patterns
        patterns['On_Balance_Volume_Divergence'] = self._detect_obv_divergence(df)
        patterns['Volume_Price_Divergence'] = (price_change > 0) & (volume < volume_sma * 0.7)
        patterns['Volume_Confirmation'] = (np.abs(price_change) > price_change.std()) & (volume > volume_sma * 1.5)
        patterns['Accumulation_Distribution'] = self._detect_accumulation_distribution(df)
        patterns['Volume_Breakout'] = self._detect_volume_breakout(df)
        patterns['Volume_Exhaustion'] = self._detect_volume_exhaustion(df)
        
        # Price-Volume Trend (PVT) patterns
        patterns['PVT_Bullish_Divergence'] = self._detect_pvt_divergence(df, 'bullish')
        patterns['PVT_Bearish_Divergence'] = self._detect_pvt_divergence(df, 'bearish')
        
        # Volume oscillator patterns
        patterns['Volume_Oscillator_Overbought'] = self._detect_volume_oscillator_extremes(df, 'overbought')
        patterns['Volume_Oscillator_Oversold'] = self._detect_volume_oscillator_extremes(df, 'oversold')
        
        return patterns
        
    def _detect_obv_divergence(self, df):
        """Detect On-Balance Volume divergence"""
        price_change = df['Close'].pct_change()
        obv = (df['Volume'] * np.sign(price_change)).cumsum()
        
        # Simplified divergence detection
        price_trend = df['Close'].rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 20 else 0)
        obv_trend = obv.rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 20 else 0)
        
        # Divergence when price and OBV trends have opposite signs
        return (price_trend > 0) & (obv_trend < 0) | (price_trend < 0) & (obv_trend > 0)
        
    def _detect_accumulation_distribution(self, df):
        """Detect accumulation/distribution patterns"""
        # Money Flow Multiplier
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        money_flow_volume = clv * df['Volume']
        ad_line = money_flow_volume.cumsum()
        
        # Accumulation when AD line is rising
        ad_trend = ad_line.rolling(10).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 10 else 0)
        return ad_trend > 0
        
    def _detect_volume_breakout(self, df):
        """Detect volume breakout patterns"""
        volume_avg = df['Volume'].rolling(50).mean()
        price_breakout = (df['Close'] > df['Close'].rolling(20).max().shift()) | (df['Close'] < df['Close'].rolling(20).min().shift())
        volume_confirmation = df['Volume'] > volume_avg * 1.5
        
        return price_breakout & volume_confirmation
        
    def _detect_volume_exhaustion(self, df):
        """Detect volume exhaustion patterns"""
        price_change = df['Close'].pct_change()
        volume_change = df['Volume'].pct_change()
        
        # High price movement with decreasing volume
        high_price_move = np.abs(price_change) > price_change.std() * 1.5
        decreasing_volume = volume_change < -0.2
        
        return high_price_move & decreasing_volume
        
    def _detect_pvt_divergence(self, df, direction):
        """Detect Price Volume Trend divergence"""
        price_change = df['Close'].pct_change()
        pvt = (price_change * df['Volume']).cumsum()
        
        price_trend = df['Close'].rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 20 else 0)
        pvt_trend = pvt.rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 20 else 0)
        
        if direction == 'bullish':
            return (price_trend < 0) & (pvt_trend > 0)
        else:  # bearish
            return (price_trend > 0) & (pvt_trend < 0)
            
    def _detect_volume_oscillator_extremes(self, df, extreme_type):
        """Detect volume oscillator extremes"""
        short_vol_avg = df['Volume'].rolling(10).mean()
        long_vol_avg = df['Volume'].rolling(30).mean()
        vol_oscillator = (short_vol_avg - long_vol_avg) / long_vol_avg * 100
        
        if extreme_type == 'overbought':
            return vol_oscillator > vol_oscillator.rolling(50).quantile(0.9)
        else:  # oversold
            return vol_oscillator < vol_oscillator.rolling(50).quantile(0.1)
            
    def _calculate_pattern_success_rate(self, df, pattern_series, timeframe):
        """Calculate success rate for a pattern"""
        try:
            if not isinstance(pattern_series, pd.Series) or not pattern_series.any():
                return 0.0
                
            pattern_dates = pattern_series[pattern_series].index
            if len(pattern_dates) == 0:
                return 0.0
                
            success_count = 0
            total_count = 0
            
            # Define success criteria based on timeframe
            if timeframe in ['5m', '15m']:
                future_periods = 12  # Look ahead 12 periods
                success_threshold = 0.005  # 0.5% move
            elif timeframe == '1h':
                future_periods = 6
                success_threshold = 0.01
            elif timeframe == '1d':
                future_periods = 5
                success_threshold = 0.02
            else:  # 1wk
                future_periods = 4
                success_threshold = 0.03
                
            for date in pattern_dates:
                try:
                    pattern_idx = df.index.get_loc(date)
                    if pattern_idx < len(df) - future_periods:
                        current_price = df['Close'].iloc[pattern_idx]
                        future_price = df['Close'].iloc[pattern_idx + future_periods]
                        
                        price_change = (future_price - current_price) / current_price
                        
                        # Consider it successful if price moves significantly in either direction
                        if abs(price_change) >= success_threshold:
                            success_count += 1
                        total_count += 1
                        
                except Exception:
                    continue
                    
            return (success_count / total_count) if total_count > 0 else 0.0
            
        except Exception:
            return 0.0
            
    def track_pattern_evolution(self, df, patterns, timeframe):
        """Track pattern evolution and lifecycle"""
        evolution = {}
        
        for pattern_name, pattern_series in patterns.items():
            if isinstance(pattern_series, pd.Series) and pattern_series.any() and not pattern_name.endswith('_success_rate'):
                evolution[pattern_name] = {
                    'total_occurrences': int(pattern_series.sum()),
                    'recent_activity': int(pattern_series.iloc[-20:].sum()),
                    'first_occurrence': pattern_series[pattern_series].index[0].strftime('%Y-%m-%d %H:%M:%S') if pattern_series.any() else None,
                    'last_occurrence': pattern_series[pattern_series].index[-1].strftime('%Y-%m-%d %H:%M:%S') if pattern_series.any() else None,
                    'frequency_trend': self._calculate_frequency_trend(pattern_series),
                    'success_rate': patterns.get(f"{pattern_name}_success_rate", 0.0),
                    'average_duration': self._calculate_average_duration(pattern_series),
                    'seasonality': self._calculate_pattern_seasonality(pattern_series, timeframe)
                }
                
        return evolution
        
    def _calculate_frequency_trend(self, pattern_series):
        """Calculate if pattern frequency is increasing or decreasing"""
        if not pattern_series.any():
            return 'stable'
            
        # Split into two halves and compare
        mid_point = len(pattern_series) // 2
        first_half = pattern_series.iloc[:mid_point].sum()
        second_half = pattern_series.iloc[mid_point:].sum()
        
        if second_half > first_half * 1.2:
            return 'increasing'
        elif second_half < first_half * 0.8:
            return 'decreasing'
        else:
            return 'stable'
            
    def _calculate_average_duration(self, pattern_series):
        """Calculate average duration of pattern occurrences"""
        if not pattern_series.any():
            return 0
            
        # Find consecutive True values
        pattern_groups = (pattern_series != pattern_series.shift()).cumsum()
        durations = pattern_series.groupby(pattern_groups).sum()
        active_durations = durations[durations > 0]
        
        return float(active_durations.mean()) if len(active_durations) > 0 else 1.0
        
    def _calculate_pattern_seasonality(self, pattern_series, timeframe):
        """Calculate pattern seasonality"""
        if not pattern_series.any():
            return {}
            
        # Add time components
        df_temp = pd.DataFrame({'pattern': pattern_series})
        df_temp['hour'] = df_temp.index.hour
        df_temp['day_of_week'] = df_temp.index.dayofweek
        df_temp['month'] = df_temp.index.month
        
        seasonality = {}
        
        if timeframe in ['5m', '15m', '1h']:
            # Hourly seasonality
            hourly_counts = df_temp.groupby('hour')['pattern'].sum()
            if hourly_counts.sum() > 0:
                seasonality['best_hour'] = int(hourly_counts.idxmax()) if hourly_counts.max() > 0 else None
                seasonality['worst_hour'] = int(hourly_counts.idxmin()) if hourly_counts.min() < hourly_counts.max() else None
                
        # Daily seasonality
        daily_counts = df_temp.groupby('day_of_week')['pattern'].sum()
        if daily_counts.sum() > 0:
            seasonality['best_day'] = int(daily_counts.idxmax()) if daily_counts.max() > 0 else None
            seasonality['worst_day'] = int(daily_counts.idxmin()) if daily_counts.min() < daily_counts.max() else None
            
        # Monthly seasonality
        monthly_counts = df_temp.groupby('month')['pattern'].sum()
        if monthly_counts.sum() > 0:
            seasonality['best_month'] = int(monthly_counts.idxmax()) if monthly_counts.max() > 0 else None
            seasonality['worst_month'] = int(monthly_counts.idxmin()) if monthly_counts.min() < monthly_counts.max() else None
            
        return seasonality

def main():
    """Main execution function"""
    print("🚀 Starting Complete Multi-Timeframe Analysis...")
    
    # Create analyzer
    analyzer = CompleteMultiTimeframeAnalyzer("^NSEI")
    
    # Fetch data for all timeframes
    analyzer.fetch_multi_timeframe_data()
    
    # Analyze all timeframes
    analyzer.analyze_all_timeframes()
    
    print("✅ Multi-timeframe analysis complete!")

if __name__ == "__main__":
    main()