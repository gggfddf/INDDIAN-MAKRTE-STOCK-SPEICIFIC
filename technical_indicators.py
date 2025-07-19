import pandas as pd
import numpy as np
import talib
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedTechnicalIndicators:
    """
    Comprehensive technical indicators with advanced pattern recognition
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.close = data['Close'].values
        self.high = data['High'].values
        self.low = data['Low'].values
        self.open = data['Open'].values
        self.volume = data['Volume'].values
        self.indicators = {}
        self.patterns = {}
        
    def calculate_all_indicators(self):
        """Calculate all 40+ technical indicators"""
        
        # 1. Moving Averages Family
        self._calculate_moving_averages()
        
        # 2. Bollinger Bands Family  
        self._calculate_bollinger_patterns()
        
        # 3. VWAP Family
        self._calculate_vwap_patterns()
        
        # 4. RSI Family
        self._calculate_rsi_patterns()
        
        # 5. MACD Family
        self._calculate_macd_patterns()
        
        # 6. Stochastic Family
        self._calculate_stochastic_patterns()
        
        # 7. Volume Indicators
        self._calculate_volume_indicators()
        
        # 8. Momentum Indicators
        self._calculate_momentum_indicators()
        
        # 9. Volatility Indicators
        self._calculate_volatility_indicators()
        
        # 10. Custom Composite Indicators
        self._calculate_composite_indicators()
        
        return self.indicators
    
    def _calculate_moving_averages(self):
        """Advanced Moving Average Analysis"""
        
        # Basic MAs
        self.indicators['SMA_20'] = talib.SMA(self.close, 20)
        self.indicators['SMA_50'] = talib.SMA(self.close, 50)
        self.indicators['SMA_200'] = talib.SMA(self.close, 200)
        self.indicators['EMA_12'] = talib.EMA(self.close, 12)
        self.indicators['EMA_26'] = talib.EMA(self.close, 26)
        
        # MA Cloud Analysis
        self.indicators['MA_Cloud_Strength'] = self._calculate_ma_cloud()
        
        # Golden/Death Cross Detection
        self.indicators['Golden_Cross'] = self._detect_golden_cross()
        self.indicators['Death_Cross'] = self._detect_death_cross()
        
        # Dynamic Support/Resistance
        self.indicators['MA_Support_Resistance'] = self._ma_support_resistance()
        
        # MA Trend Strength
        self.indicators['MA_Trend_Strength'] = self._ma_trend_strength()
    
    def _calculate_bollinger_patterns(self):
        """Advanced Bollinger Bands Analysis"""
        
        # Basic Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(self.close, 20, 2, 2)
        self.indicators['BB_Upper'] = bb_upper
        self.indicators['BB_Middle'] = bb_middle  
        self.indicators['BB_Lower'] = bb_lower
        
        # Bollinger Band Width
        self.indicators['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        
        # Bollinger Squeeze Detection
        self.indicators['BB_Squeeze'] = self._detect_bb_squeeze()
        
        # Bollinger Expansion Patterns
        self.indicators['BB_Expansion'] = self._detect_bb_expansion()
        
        # Band Walking Detection
        self.indicators['BB_Walking'] = self._detect_band_walking()
        
        # %B Indicator
        self.indicators['BB_Percent'] = (self.close - bb_lower) / (bb_upper - bb_lower)
    
    def _calculate_vwap_patterns(self):
        """Advanced VWAP Analysis"""
        
        # Calculate VWAP
        self.indicators['VWAP'] = self._calculate_vwap()
        
        # VWAP Flatness Detection
        self.indicators['VWAP_Flat'] = self._detect_vwap_flatness()
        
        # VWAP Reversion Patterns
        self.indicators['VWAP_Reversion'] = self._vwap_reversion_signals()
        
        # Multi-timeframe VWAP
        self.indicators['VWAP_MTF'] = self._multi_timeframe_vwap()
        
        # VWAP Distance
        self.indicators['VWAP_Distance'] = (self.close - self.indicators['VWAP']) / self.indicators['VWAP'] * 100
    
    def _calculate_rsi_patterns(self):
        """Advanced RSI Analysis"""
        
        # Basic RSI
        self.indicators['RSI_14'] = talib.RSI(self.close, 14)
        
        # RSI Divergence Detection
        self.indicators['RSI_Divergence'] = self._detect_rsi_divergence()
        
        # RSI Trend Analysis
        self.indicators['RSI_Trend'] = self._rsi_trend_analysis()
        
        # Overbought/Oversold Clusters
        self.indicators['RSI_OB_OS_Clusters'] = self._rsi_ob_os_clusters()
        
        # RSI Mean Reversion
        self.indicators['RSI_Mean_Reversion'] = self._rsi_mean_reversion()
    
    def _calculate_macd_patterns(self):
        """Advanced MACD Analysis"""
        
        # Basic MACD
        macd, macd_signal, macd_hist = talib.MACD(self.close, 12, 26, 9)
        self.indicators['MACD'] = macd
        self.indicators['MACD_Signal'] = macd_signal
        self.indicators['MACD_Histogram'] = macd_hist
        
        # MACD Histogram Patterns
        self.indicators['MACD_Hist_Patterns'] = self._macd_histogram_patterns()
        
        # MACD Signal Line Analysis
        self.indicators['MACD_Signal_Analysis'] = self._macd_signal_analysis()
        
        # MACD Trend Strength
        self.indicators['MACD_Trend_Strength'] = self._macd_trend_strength()
        
        # MACD Zero Line Analysis
        self.indicators['MACD_Zero_Cross'] = self._macd_zero_cross()
    
    def _calculate_stochastic_patterns(self):
        """Advanced Stochastic Analysis"""
        
        # Basic Stochastic
        slowk, slowd = talib.STOCH(self.high, self.low, self.close, 14, 3, 0, 3, 0)
        self.indicators['Stoch_K'] = slowk
        self.indicators['Stoch_D'] = slowd
        
        # K/D Crossover Patterns
        self.indicators['Stoch_Crossover'] = self._stochastic_crossover_patterns()
        
        # Stochastic Divergence
        self.indicators['Stoch_Divergence'] = self._stochastic_divergence()
        
        # Stochastic Trend Analysis
        self.indicators['Stoch_Trend'] = self._stochastic_trend_analysis()
    
    def _calculate_volume_indicators(self):
        """Advanced Volume Analysis"""
        
        # On Balance Volume
        self.indicators['OBV'] = talib.OBV(self.close, self.volume)
        
        # OBV Patterns
        self.indicators['OBV_Patterns'] = self._obv_pattern_analysis()
        
        # Volume Price Trend
        self.indicators['VPT'] = self._volume_price_trend()
        
        # Accumulation/Distribution
        self.indicators['AD'] = talib.AD(self.high, self.low, self.close, self.volume)
        
        # Volume Oscillator
        self.indicators['Volume_Oscillator'] = self._volume_oscillator()
        
        # Money Flow Index
        self.indicators['MFI'] = talib.MFI(self.high, self.low, self.close, self.volume, 14)
        
        # Volume Rate of Change
        self.indicators['Volume_ROC'] = talib.ROC(self.volume, 10)
    
    def _calculate_momentum_indicators(self):
        """Advanced Momentum Analysis"""
        
        # Rate of Change
        self.indicators['ROC_10'] = talib.ROC(self.close, 10)
        self.indicators['ROC_20'] = talib.ROC(self.close, 20)
        
        # ROC Patterns
        self.indicators['ROC_Patterns'] = self._roc_pattern_analysis()
        
        # Momentum Divergence
        self.indicators['Momentum_Divergence'] = self._momentum_divergence()
        
        # Williams %R
        self.indicators['Williams_R'] = talib.WILLR(self.high, self.low, self.close, 14)
        
        # Commodity Channel Index
        self.indicators['CCI'] = talib.CCI(self.high, self.low, self.close, 14)
        
        # Ultimate Oscillator
        self.indicators['Ultimate_Osc'] = talib.ULTOSC(self.high, self.low, self.close, 7, 14, 28)
    
    def _calculate_volatility_indicators(self):
        """Advanced Volatility Analysis"""
        
        # Average True Range
        self.indicators['ATR'] = talib.ATR(self.high, self.low, self.close, 14)
        
        # ATR Patterns
        self.indicators['ATR_Patterns'] = self._atr_pattern_analysis()
        
        # Volatility Breakout Signals
        self.indicators['Vol_Breakout'] = self._volatility_breakout_signals()
        
        # True Range
        self.indicators['TRANGE'] = talib.TRANGE(self.high, self.low, self.close)
        
        # Normalized ATR
        self.indicators['ATR_Normalized'] = self.indicators['ATR'] / self.close * 100
    
    def _calculate_composite_indicators(self):
        """Custom Composite Indicators"""
        
        # Multi-Indicator Trend Score
        self.indicators['Trend_Score'] = self._calculate_trend_score()
        
        # Momentum Composite
        self.indicators['Momentum_Composite'] = self._momentum_composite()
        
        # Volatility Regime
        self.indicators['Volatility_Regime'] = self._volatility_regime()
        
        # Market Strength Index
        self.indicators['Market_Strength'] = self._market_strength_index()
        
        # Reversal Probability
        self.indicators['Reversal_Probability'] = self._reversal_probability()
    
    # Helper methods for pattern detection
    
    def _calculate_ma_cloud(self):
        """Calculate MA Cloud Strength"""
        ma_cloud = np.zeros(len(self.close))
        sma20 = self.indicators['SMA_20']
        sma50 = self.indicators['SMA_50']
        sma200 = self.indicators['SMA_200']
        
        for i in range(len(self.close)):
            if pd.isna(sma20[i]) or pd.isna(sma50[i]) or pd.isna(sma200[i]):
                ma_cloud[i] = 0
            else:
                # Calculate alignment strength
                if sma20[i] > sma50[i] > sma200[i]:
                    ma_cloud[i] = 1  # Strong bullish
                elif sma20[i] > sma50[i] and sma50[i] < sma200[i]:
                    ma_cloud[i] = 0.5  # Weak bullish
                elif sma20[i] < sma50[i] < sma200[i]:
                    ma_cloud[i] = -1  # Strong bearish
                elif sma20[i] < sma50[i] and sma50[i] > sma200[i]:
                    ma_cloud[i] = -0.5  # Weak bearish
                else:
                    ma_cloud[i] = 0  # Neutral
        
        return ma_cloud
    
    def _detect_golden_cross(self):
        """Detect Golden Cross patterns"""
        sma50 = self.indicators['SMA_50']
        sma200 = self.indicators['SMA_200']
        
        golden_cross = np.zeros(len(self.close))
        for i in range(1, len(self.close)):
            if (sma50[i] > sma200[i] and sma50[i-1] <= sma200[i-1] and 
                not pd.isna(sma50[i]) and not pd.isna(sma200[i])):
                golden_cross[i] = 1
        
        return golden_cross
    
    def _detect_death_cross(self):
        """Detect Death Cross patterns"""
        sma50 = self.indicators['SMA_50']
        sma200 = self.indicators['SMA_200']
        
        death_cross = np.zeros(len(self.close))
        for i in range(1, len(self.close)):
            if (sma50[i] < sma200[i] and sma50[i-1] >= sma200[i-1] and 
                not pd.isna(sma50[i]) and not pd.isna(sma200[i])):
                death_cross[i] = 1
        
        return death_cross
    
    def _ma_support_resistance(self):
        """Calculate MA-based support/resistance"""
        support_resistance = np.zeros(len(self.close))
        sma20 = self.indicators['SMA_20']
        
        for i in range(1, len(self.close)):
            if not pd.isna(sma20[i]):
                # Check if price bounced off MA
                if (self.low[i] <= sma20[i] <= self.high[i] and 
                    self.close[i] > sma20[i]):
                    support_resistance[i] = 1  # Support
                elif (self.low[i] <= sma20[i] <= self.high[i] and 
                      self.close[i] < sma20[i]):
                    support_resistance[i] = -1  # Resistance
        
        return support_resistance
    
    def _ma_trend_strength(self):
        """Calculate MA trend strength"""
        trend_strength = np.zeros(len(self.close))
        sma20 = self.indicators['SMA_20']
        
        for i in range(20, len(self.close)):
            if not pd.isna(sma20[i]):
                # Calculate slope of MA
                slope = (sma20[i] - sma20[i-5]) / sma20[i-5] * 100
                trend_strength[i] = slope
        
        return trend_strength
    
    def _detect_bb_squeeze(self):
        """Detect Bollinger Band Squeeze"""
        bb_width = self.indicators['BB_Width']
        squeeze = np.zeros(len(self.close))
        
        # Use rolling minimum to detect squeeze
        window = 20
        for i in range(window, len(self.close)):
            if not pd.isna(bb_width[i]):
                min_width = np.nanmin(bb_width[i-window:i])
                if bb_width[i] <= min_width * 1.1:  # Within 10% of minimum
                    squeeze[i] = 1
        
        return squeeze
    
    def _detect_bb_expansion(self):
        """Detect Bollinger Band Expansion"""
        bb_width = self.indicators['BB_Width']
        expansion = np.zeros(len(self.close))
        
        for i in range(1, len(self.close)):
            if not pd.isna(bb_width[i]) and not pd.isna(bb_width[i-1]):
                if bb_width[i] > bb_width[i-1] * 1.2:  # 20% expansion
                    expansion[i] = 1
        
        return expansion
    
    def _detect_band_walking(self):
        """Detect Band Walking patterns"""
        bb_percent = self.indicators['BB_Percent']
        walking = np.zeros(len(self.close))
        
        window = 5
        for i in range(window, len(self.close)):
            if not pd.isna(bb_percent[i]):
                recent_vals = bb_percent[i-window:i+1]
                if np.all(recent_vals > 0.8):  # Upper band walking
                    walking[i] = 1
                elif np.all(recent_vals < 0.2):  # Lower band walking
                    walking[i] = -1
        
        return walking
    
    def _calculate_vwap(self):
        """Calculate VWAP"""
        typical_price = (self.high + self.low + self.close) / 3
        vwap = np.cumsum(typical_price * self.volume) / np.cumsum(self.volume)
        return vwap
    
    def _detect_vwap_flatness(self):
        """Detect VWAP flatness significance"""
        vwap = self.indicators['VWAP']
        flatness = np.zeros(len(self.close))
        
        window = 10
        for i in range(window, len(self.close)):
            if not pd.isna(vwap[i]):
                vwap_range = np.nanmax(vwap[i-window:i+1]) - np.nanmin(vwap[i-window:i+1])
                price_range = np.nanmax(self.close[i-window:i+1]) - np.nanmin(self.close[i-window:i+1])
                if vwap_range / price_range < 0.1:  # VWAP is flat relative to price
                    flatness[i] = 1
        
        return flatness
    
    def _vwap_reversion_signals(self):
        """VWAP reversion pattern detection"""
        vwap = self.indicators['VWAP']
        reversion = np.zeros(len(self.close))
        
        for i in range(2, len(self.close)):
            if not pd.isna(vwap[i]):
                # Check for price moving away from VWAP then reverting
                if (self.close[i-2] > vwap[i-2] and 
                    self.close[i-1] > vwap[i-1] and 
                    self.close[i] < vwap[i]):
                    reversion[i] = -1  # Bearish reversion
                elif (self.close[i-2] < vwap[i-2] and 
                      self.close[i-1] < vwap[i-1] and 
                      self.close[i] > vwap[i]):
                    reversion[i] = 1  # Bullish reversion
        
        return reversion
    
    def _multi_timeframe_vwap(self):
        """Multi-timeframe VWAP analysis"""
        # This is a simplified version - in practice, you'd need actual MTF data
        vwap = self.indicators['VWAP']
        mtf_analysis = np.zeros(len(self.close))
        
        # Calculate VWAP alignment across different periods
        for i in range(50, len(self.close)):
            if not pd.isna(vwap[i]):
                short_vwap = np.mean(vwap[i-10:i+1])
                long_vwap = np.mean(vwap[i-50:i+1])
                
                if self.close[i] > short_vwap > long_vwap:
                    mtf_analysis[i] = 1  # Bullish alignment
                elif self.close[i] < short_vwap < long_vwap:
                    mtf_analysis[i] = -1  # Bearish alignment
        
        return mtf_analysis
    
    def _detect_rsi_divergence(self):
        """Detect RSI divergence patterns"""
        rsi = self.indicators['RSI_14']
        divergence = np.zeros(len(self.close))
        
        window = 20
        for i in range(window, len(self.close)):
            if not pd.isna(rsi[i]):
                # Find recent peaks and troughs
                price_peak = np.argmax(self.close[i-window:i+1]) + i - window
                rsi_peak = np.argmax(rsi[i-window:i+1]) + i - window
                
                price_trough = np.argmin(self.close[i-window:i+1]) + i - window
                rsi_trough = np.argmin(rsi[i-window:i+1]) + i - window
                
                # Check for bearish divergence
                if (price_peak == i and self.close[i] > self.close[price_peak-window] and 
                    rsi[i] < rsi[rsi_peak]):
                    divergence[i] = -1
                
                # Check for bullish divergence
                if (price_trough == i and self.close[i] < self.close[price_trough-window] and 
                    rsi[i] > rsi[rsi_trough]):
                    divergence[i] = 1
        
        return divergence
    
    def _rsi_trend_analysis(self):
        """RSI trend analysis"""
        rsi = self.indicators['RSI_14']
        trend = np.zeros(len(self.close))
        
        window = 10
        for i in range(window, len(self.close)):
            if not pd.isna(rsi[i]):
                # Calculate RSI trend
                rsi_slope = (rsi[i] - rsi[i-window]) / window
                if rsi_slope > 2:
                    trend[i] = 1  # Strong bullish RSI trend
                elif rsi_slope < -2:
                    trend[i] = -1  # Strong bearish RSI trend
        
        return trend
    
    def _rsi_ob_os_clusters(self):
        """RSI overbought/oversold cluster analysis"""
        rsi = self.indicators['RSI_14']
        clusters = np.zeros(len(self.close))
        
        for i in range(5, len(self.close)):
            if not pd.isna(rsi[i]):
                recent_rsi = rsi[i-5:i+1]
                if np.sum(recent_rsi > 70) >= 3:  # Overbought cluster
                    clusters[i] = 1
                elif np.sum(recent_rsi < 30) >= 3:  # Oversold cluster
                    clusters[i] = -1
        
        return clusters
    
    def _rsi_mean_reversion(self):
        """RSI mean reversion signals"""
        rsi = self.indicators['RSI_14']
        reversion = np.zeros(len(self.close))
        
        for i in range(1, len(self.close)):
            if not pd.isna(rsi[i]):
                if rsi[i-1] > 70 and rsi[i] < 70:  # Exit overbought
                    reversion[i] = -1
                elif rsi[i-1] < 30 and rsi[i] > 30:  # Exit oversold
                    reversion[i] = 1
        
        return reversion
    
    # Continue with more pattern detection methods...
    
    def _macd_histogram_patterns(self):
        """MACD Histogram pattern analysis"""
        hist = self.indicators['MACD_Histogram']
        patterns = np.zeros(len(self.close))
        
        for i in range(3, len(self.close)):
            if not pd.isna(hist[i]):
                # Detect histogram reversal patterns
                if (hist[i-2] < hist[i-1] and hist[i-1] > hist[i] and hist[i-1] > 0):
                    patterns[i] = -1  # Bearish histogram divergence
                elif (hist[i-2] > hist[i-1] and hist[i-1] < hist[i] and hist[i-1] < 0):
                    patterns[i] = 1  # Bullish histogram divergence
        
        return patterns
    
    def _macd_signal_analysis(self):
        """MACD signal line analysis"""
        macd = self.indicators['MACD']
        signal = self.indicators['MACD_Signal']
        analysis = np.zeros(len(self.close))
        
        for i in range(1, len(self.close)):
            if not pd.isna(macd[i]) and not pd.isna(signal[i]):
                # MACD crossover signals
                if macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
                    analysis[i] = 1  # Bullish crossover
                elif macd[i] < signal[i] and macd[i-1] >= signal[i-1]:
                    analysis[i] = -1  # Bearish crossover
        
        return analysis
    
    def _macd_trend_strength(self):
        """MACD trend strength analysis"""
        macd = self.indicators['MACD']
        strength = np.zeros(len(self.close))
        
        window = 10
        for i in range(window, len(self.close)):
            if not pd.isna(macd[i]):
                macd_slope = (macd[i] - macd[i-window]) / window
                strength[i] = macd_slope
        
        return strength
    
    def _macd_zero_cross(self):
        """MACD zero line cross analysis"""
        macd = self.indicators['MACD']
        zero_cross = np.zeros(len(self.close))
        
        for i in range(1, len(self.close)):
            if not pd.isna(macd[i]):
                if macd[i] > 0 and macd[i-1] <= 0:
                    zero_cross[i] = 1  # Bullish zero cross
                elif macd[i] < 0 and macd[i-1] >= 0:
                    zero_cross[i] = -1  # Bearish zero cross
        
        return zero_cross
    
    def _stochastic_crossover_patterns(self):
        """Stochastic K/D crossover patterns"""
        k = self.indicators['Stoch_K']
        d = self.indicators['Stoch_D']
        patterns = np.zeros(len(self.close))
        
        for i in range(1, len(self.close)):
            if not pd.isna(k[i]) and not pd.isna(d[i]):
                if k[i] > d[i] and k[i-1] <= d[i-1]:
                    patterns[i] = 1  # Bullish crossover
                elif k[i] < d[i] and k[i-1] >= d[i-1]:
                    patterns[i] = -1  # Bearish crossover
        
        return patterns
    
    def _stochastic_divergence(self):
        """Stochastic divergence detection"""
        k = self.indicators['Stoch_K']
        divergence = np.zeros(len(self.close))
        
        window = 15
        for i in range(window, len(self.close)):
            if not pd.isna(k[i]):
                # Similar to RSI divergence but for Stochastic
                price_high = np.max(self.close[i-window:i+1])
                stoch_high = np.max(k[i-window:i+1])
                
                if (self.close[i] == price_high and k[i] < stoch_high):
                    divergence[i] = -1  # Bearish divergence
                
                price_low = np.min(self.close[i-window:i+1])
                stoch_low = np.min(k[i-window:i+1])
                
                if (self.close[i] == price_low and k[i] > stoch_low):
                    divergence[i] = 1  # Bullish divergence
        
        return divergence
    
    def _stochastic_trend_analysis(self):
        """Stochastic trend analysis"""
        k = self.indicators['Stoch_K']
        trend = np.zeros(len(self.close))
        
        window = 5
        for i in range(window, len(self.close)):
            if not pd.isna(k[i]):
                recent_k = k[i-window:i+1]
                if np.all(recent_k > 80):
                    trend[i] = 1  # Strong bullish
                elif np.all(recent_k < 20):
                    trend[i] = -1  # Strong bearish
        
        return trend
    
    def _obv_pattern_analysis(self):
        """OBV pattern analysis"""
        obv = self.indicators['OBV']
        patterns = np.zeros(len(self.close))
        
        window = 20
        for i in range(window, len(self.close)):
            if not pd.isna(obv[i]):
                # OBV trend vs price trend
                obv_slope = (obv[i] - obv[i-window]) / window
                price_slope = (self.close[i] - self.close[i-window]) / window
                
                if obv_slope > 0 and price_slope < 0:
                    patterns[i] = 1  # Bullish divergence
                elif obv_slope < 0 and price_slope > 0:
                    patterns[i] = -1  # Bearish divergence
        
        return patterns
    
    def _volume_price_trend(self):
        """Volume Price Trend calculation"""
        vpt = np.zeros(len(self.close))
        vpt[0] = 0
        
        for i in range(1, len(self.close)):
            if self.close[i-1] != 0:
                price_change = (self.close[i] - self.close[i-1]) / self.close[i-1]
                vpt[i] = vpt[i-1] + (self.volume[i] * price_change)
        
        return vpt
    
    def _volume_oscillator(self):
        """Volume Oscillator calculation"""
        vol_short = talib.SMA(self.volume.astype(float), 5)
        vol_long = talib.SMA(self.volume.astype(float), 20)
        
        vol_osc = np.zeros(len(self.close))
        for i in range(len(self.close)):
            if not pd.isna(vol_long[i]) and vol_long[i] != 0:
                vol_osc[i] = ((vol_short[i] - vol_long[i]) / vol_long[i]) * 100
        
        return vol_osc
    
    def _roc_pattern_analysis(self):
        """Rate of Change pattern analysis"""
        roc10 = self.indicators['ROC_10']
        roc20 = self.indicators['ROC_20']
        patterns = np.zeros(len(self.close))
        
        for i in range(1, len(self.close)):
            if not pd.isna(roc10[i]) and not pd.isna(roc20[i]):
                # ROC crossover patterns
                if roc10[i] > roc20[i] and roc10[i-1] <= roc20[i-1]:
                    patterns[i] = 1  # Bullish momentum
                elif roc10[i] < roc20[i] and roc10[i-1] >= roc20[i-1]:
                    patterns[i] = -1  # Bearish momentum
        
        return patterns
    
    def _momentum_divergence(self):
        """Momentum divergence analysis"""
        roc = self.indicators['ROC_10']
        divergence = np.zeros(len(self.close))
        
        window = 15
        for i in range(window, len(self.close)):
            if not pd.isna(roc[i]):
                # Price vs momentum divergence
                price_change = self.close[i] - self.close[i-window]
                roc_change = roc[i] - roc[i-window]
                
                if price_change > 0 and roc_change < 0:
                    divergence[i] = -1  # Bearish divergence
                elif price_change < 0 and roc_change > 0:
                    divergence[i] = 1  # Bullish divergence
        
        return divergence
    
    def _atr_pattern_analysis(self):
        """ATR pattern analysis"""
        atr = self.indicators['ATR']
        patterns = np.zeros(len(self.close))
        
        window = 10
        for i in range(window, len(self.close)):
            if not pd.isna(atr[i]):
                atr_ma = np.nanmean(atr[i-window:i])
                if atr[i] > atr_ma * 1.5:
                    patterns[i] = 1  # High volatility
                elif atr[i] < atr_ma * 0.5:
                    patterns[i] = -1  # Low volatility
        
        return patterns
    
    def _volatility_breakout_signals(self):
        """Volatility breakout signal detection"""
        atr = self.indicators['ATR']
        signals = np.zeros(len(self.close))
        
        window = 20
        for i in range(window, len(self.close)):
            if not pd.isna(atr[i]):
                avg_atr = np.nanmean(atr[i-window:i])
                price_move = abs(self.close[i] - self.close[i-1])
                
                if price_move > avg_atr * 2:
                    signals[i] = 1  # Significant breakout
        
        return signals
    
    def _calculate_trend_score(self):
        """Multi-indicator trend score"""
        score = np.zeros(len(self.close))
        
        # Components to include in trend score
        components = []
        if 'MA_Cloud_Strength' in self.indicators:
            components.append(self.indicators['MA_Cloud_Strength'])
        if 'RSI_Trend' in self.indicators:
            components.append(self.indicators['RSI_Trend'])
        if 'MACD_Trend_Strength' in self.indicators:
            components.append(np.sign(self.indicators['MACD_Trend_Strength']))
        
        if components:
            score = np.mean(components, axis=0)
        
        return score
    
    def _momentum_composite(self):
        """Composite momentum indicator"""
        momentum = np.zeros(len(self.close))
        
        # Combine multiple momentum indicators
        components = []
        if 'ROC_10' in self.indicators:
            components.append(np.sign(self.indicators['ROC_10']))
        if 'RSI_14' in self.indicators:
            rsi_momentum = (self.indicators['RSI_14'] - 50) / 50
            components.append(rsi_momentum)
        if 'MACD' in self.indicators:
            components.append(np.sign(self.indicators['MACD']))
        
        if components:
            momentum = np.mean(components, axis=0)
        
        return momentum
    
    def _volatility_regime(self):
        """Volatility regime classification"""
        atr = self.indicators['ATR']
        regime = np.zeros(len(self.close))
        
        window = 50
        for i in range(window, len(self.close)):
            if not pd.isna(atr[i]):
                atr_percentile = stats.percentileofscore(atr[i-window:i], atr[i])
                
                if atr_percentile > 80:
                    regime[i] = 2  # High volatility
                elif atr_percentile > 60:
                    regime[i] = 1  # Medium-high volatility
                elif atr_percentile < 20:
                    regime[i] = -2  # Low volatility
                elif atr_percentile < 40:
                    regime[i] = -1  # Medium-low volatility
                else:
                    regime[i] = 0  # Normal volatility
        
        return regime
    
    def _market_strength_index(self):
        """Market strength composite index"""
        strength = np.zeros(len(self.close))
        
        # Combine volume, momentum, and trend indicators
        components = []
        if 'Volume_Oscillator' in self.indicators:
            components.append(np.sign(self.indicators['Volume_Oscillator']))
        if 'Trend_Score' in self.indicators:
            components.append(self.indicators['Trend_Score'])
        if 'Momentum_Composite' in self.indicators:
            components.append(self.indicators['Momentum_Composite'])
        
        if components:
            strength = np.mean(components, axis=0)
        
        return strength
    
    def _reversal_probability(self):
        """Calculate reversal probability"""
        probability = np.zeros(len(self.close))
        
        # Factors that indicate potential reversal
        reversal_factors = []
        
        if 'RSI_Divergence' in self.indicators:
            reversal_factors.append(self.indicators['RSI_Divergence'])
        if 'MACD_Hist_Patterns' in self.indicators:
            reversal_factors.append(self.indicators['MACD_Hist_Patterns'])
        if 'Stoch_Divergence' in self.indicators:
            reversal_factors.append(self.indicators['Stoch_Divergence'])
        if 'BB_Walking' in self.indicators:
            reversal_factors.append(-self.indicators['BB_Walking'])  # Inverse for reversal
        
        if reversal_factors:
            probability = np.mean([np.abs(factor) for factor in reversal_factors], axis=0)
        
        return probability
    
    def get_current_signals(self):
        """Get current signals from all indicators"""
        signals = {}
        latest_idx = -1
        
        for name, values in self.indicators.items():
            if len(values) > 0 and not pd.isna(values[latest_idx]):
                signals[name] = float(values[latest_idx])
        
        return signals
    
    def get_pattern_summary(self):
        """Get summary of detected patterns"""
        patterns = {}
        latest_idx = -1
        
        pattern_indicators = [
            'Golden_Cross', 'Death_Cross', 'BB_Squeeze', 'BB_Expansion',
            'BB_Walking', 'VWAP_Flat', 'VWAP_Reversion', 'RSI_Divergence',
            'MACD_Hist_Patterns', 'Stoch_Crossover', 'Vol_Breakout'
        ]
        
        for pattern in pattern_indicators:
            if pattern in self.indicators:
                value = self.indicators[pattern][latest_idx]
                if not pd.isna(value) and value != 0:
                    patterns[pattern] = float(value)
        
        return patterns