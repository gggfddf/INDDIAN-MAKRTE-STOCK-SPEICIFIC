import pandas as pd
import numpy as np
import talib
from scipy import signal, stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import cv2
import warnings
warnings.filterwarnings('ignore')

class AdvancedPriceActionAnalysis:
    """
    Comprehensive price action analysis with traditional and ML-discovered patterns
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.close = data['Close'].values
        self.high = data['High'].values
        self.low = data['Low'].values
        self.open = data['Open'].values
        self.volume = data['Volume'].values
        self.candlestick_patterns = {}
        self.chart_patterns = {}
        self.ml_patterns = {}
        self.support_resistance = {}
        
    def analyze_all_patterns(self):
        """Analyze all price action patterns"""
        
        # 1. Traditional Candlestick Patterns
        self._detect_traditional_candlesticks()
        
        # 2. ML-Discovered Candlestick Patterns
        self._discover_ml_candlestick_patterns()
        
        # 3. Chart Pattern Analysis
        self._detect_chart_patterns()
        
        # 4. Support/Resistance Analysis
        self._analyze_support_resistance()
        
        # 5. Gap Analysis
        self._analyze_gaps()
        
        # 6. Volume Price Analysis
        self._analyze_volume_price_patterns()
        
        # 7. Breakout Analysis
        self._analyze_breakouts()
        
        # 8. Reversal Analysis
        self._analyze_reversals()
        
        return {
            'candlestick_patterns': self.candlestick_patterns,
            'chart_patterns': self.chart_patterns,
            'ml_patterns': self.ml_patterns,
            'support_resistance': self.support_resistance
        }
    
    def _detect_traditional_candlesticks(self):
        """Detect all traditional candlestick patterns"""
        
        # Single candlestick patterns
        self.candlestick_patterns['Doji'] = talib.CDLDOJI(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Hammer'] = talib.CDLHAMMER(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Hanging_Man'] = talib.CDLHANGINGMAN(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Inverted_Hammer'] = talib.CDLINVERTEDHAMMER(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Shooting_Star'] = talib.CDLSHOOTINGSTAR(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Marubozu'] = talib.CDLMARUBOZU(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Spinning_Top'] = talib.CDLSPINNINGTOP(self.open, self.high, self.low, self.close)
        
        # Two candlestick patterns
        self.candlestick_patterns['Engulfing'] = talib.CDLENGULFING(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Harami'] = talib.CDLHARAMI(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Piercing'] = talib.CDLPIERCING(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Dark_Cloud'] = talib.CDLDARKCLOUDCOVER(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Tweezer_Top'] = talib.CDLHIKKAKEMOD(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Tweezer_Bottom'] = talib.CDLHIKKAKE(self.open, self.high, self.low, self.close)
        
        # Three candlestick patterns
        self.candlestick_patterns['Morning_Star'] = talib.CDLMORNINGSTAR(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Evening_Star'] = talib.CDLEVENINGSTAR(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Three_White_Soldiers'] = talib.CDL3WHITESOLDIERS(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Three_Black_Crows'] = talib.CDL3BLACKCROWS(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Inside_Three'] = talib.CDL3INSIDE(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Outside_Three'] = talib.CDL3OUTSIDE(self.open, self.high, self.low, self.close)
        
        # Advanced patterns
        self.candlestick_patterns['Abandoned_Baby'] = talib.CDLABANDONEDBABY(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Belt_Hold'] = talib.CDLBELTHOLD(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Breakaway'] = talib.CDLBREAKAWAY(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Concealing_Baby_Swallow'] = talib.CDLCONCEALBABYSWALL(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Counterattack'] = talib.CDLCOUNTERATTACK(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Dragonfly_Doji'] = talib.CDLDRAGONFLYDOJI(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Gravestone_Doji'] = talib.CDLGRAVESTONEDOJI(self.open, self.high, self.low, self.close)
        self.candlestick_patterns['Long_Legged_Doji'] = talib.CDLLONGLEGGEDDOJI(self.open, self.high, self.low, self.close)
        
        # Pattern confidence scoring
        self._calculate_pattern_confidence()
    
    def _discover_ml_candlestick_patterns(self):
        """Use ML to discover new candlestick patterns"""
        
        # Feature engineering for candlestick analysis
        features = self._extract_candlestick_features()
        
        # Cluster analysis to find similar patterns
        self._cluster_candlestick_patterns(features)
        
        # Anomaly detection for unusual patterns
        self._detect_anomalous_patterns(features)
        
        # Pattern mining for sequence discovery
        self._mine_candlestick_sequences()
        
    def _extract_candlestick_features(self):
        """Extract comprehensive features from candlestick data"""
        features = []
        
        for i in range(len(self.close)):
            if i < 2:  # Need at least 2 previous candles
                features.append([0] * 25)
                continue
                
            # Current candle features
            body_size = abs(self.close[i] - self.open[i])
            upper_shadow = self.high[i] - max(self.open[i], self.close[i])
            lower_shadow = min(self.open[i], self.close[i]) - self.low[i]
            range_size = self.high[i] - self.low[i]
            
            # Ratios
            body_ratio = body_size / range_size if range_size > 0 else 0
            upper_shadow_ratio = upper_shadow / range_size if range_size > 0 else 0
            lower_shadow_ratio = lower_shadow / range_size if range_size > 0 else 0
            
            # Direction
            is_bullish = 1 if self.close[i] > self.open[i] else 0
            
            # Volume features
            volume_ratio = self.volume[i] / np.mean(self.volume[max(0, i-20):i+1]) if i > 0 else 1
            
            # Previous candle comparison
            prev_body_size = abs(self.close[i-1] - self.open[i-1])
            prev_range_size = self.high[i-1] - self.low[i-1]
            body_size_ratio = body_size / prev_body_size if prev_body_size > 0 else 0
            
            # Gap analysis
            gap_up = max(0, min(self.open[i], self.close[i]) - max(self.open[i-1], self.close[i-1]))
            gap_down = max(0, min(self.open[i-1], self.close[i-1]) - max(self.open[i], self.close[i]))
            
            # Price position features
            high_position = (max(self.open[i], self.close[i]) - self.low[i]) / range_size if range_size > 0 else 0
            low_position = (self.high[i] - min(self.open[i], self.close[i])) / range_size if range_size > 0 else 0
            
            # Multi-timeframe features
            sma_position = self.close[i] / np.mean(self.close[max(0, i-20):i+1]) if i > 0 else 1
            
            feature_vector = [
                body_size, upper_shadow, lower_shadow, range_size,
                body_ratio, upper_shadow_ratio, lower_shadow_ratio,
                is_bullish, volume_ratio, body_size_ratio,
                gap_up, gap_down, high_position, low_position, sma_position,
                # Previous candle features
                prev_body_size, prev_range_size,
                # Volatility features
                np.std(self.close[max(0, i-5):i+1]) if i > 5 else 0,
                # Momentum features
                (self.close[i] - self.close[i-1]) / self.close[i-1] if i > 0 and self.close[i-1] > 0 else 0,
                # Relative strength
                np.mean([self.close[j] > self.open[j] for j in range(max(0, i-5), i+1)]) if i > 5 else 0,
                # Pattern complexity
                len(set([self.open[j], self.close[j], self.high[j], self.low[j]] for j in range(max(0, i-3), i+1))),
                # Fibonacci levels
                self._fibonacci_level(i),
                # Support/resistance proximity
                self._sr_proximity(i),
                # Time-based features
                i % 7,  # Day of week proxy
                i % 30  # Month proxy
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _cluster_candlestick_patterns(self, features):
        """Cluster candlestick patterns to find new patterns"""
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=20, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        density_clusters = dbscan.fit_predict(features_scaled)
        
        # Analyze cluster patterns
        self.ml_patterns['Cluster_Patterns'] = self._analyze_clusters(clusters, density_clusters)
        
        # Find cluster-based signals
        self.ml_patterns['Cluster_Signals'] = self._cluster_signals(clusters, features)
        
    def _detect_anomalous_patterns(self, features):
        """Detect anomalous candlestick patterns using isolation forest"""
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(features_scaled)
        
        # Mark anomalous patterns
        self.ml_patterns['Anomalous_Patterns'] = anomalies
        
        # Analyze anomaly characteristics
        self.ml_patterns['Anomaly_Analysis'] = self._analyze_anomalies(anomalies, features)
        
    def _mine_candlestick_sequences(self):
        """Mine candlestick sequences for pattern discovery"""
        
        # Create sequence representations
        sequences = self._create_candlestick_sequences()
        
        # Find frequent sequences
        frequent_sequences = self._find_frequent_sequences(sequences)
        
        # Analyze sequence predictive power
        self.ml_patterns['Sequence_Patterns'] = frequent_sequences
        self.ml_patterns['Sequence_Predictions'] = self._analyze_sequence_predictions(sequences)
        
    def _detect_chart_patterns(self):
        """Detect traditional and novel chart patterns"""
        
        # Traditional chart patterns
        self._detect_head_shoulders()
        self._detect_triangles()
        self._detect_flags_pennants()
        self._detect_channels()
        self._detect_wedges()
        
        # Novel pattern discovery using computer vision
        self._discover_cv_patterns()
        
        # Pattern completion probability
        self._calculate_pattern_completion()
        
    def _detect_head_shoulders(self):
        """Detect Head and Shoulders patterns"""
        hs_patterns = np.zeros(len(self.close))
        
        window = 20
        for i in range(window, len(self.close) - window):
            # Find local maxima
            left_peak = self._find_local_peak(i - window, i - window//2)
            head_peak = self._find_local_peak(i - window//2, i + window//2)
            right_peak = self._find_local_peak(i + window//2, i + window)
            
            if left_peak and head_peak and right_peak:
                left_idx, left_val = left_peak
                head_idx, head_val = head_peak
                right_idx, right_val = right_peak
                
                # Check H&S criteria
                if (head_val > left_val * 1.02 and head_val > right_val * 1.02 and
                    abs(left_val - right_val) / left_val < 0.05):  # Similar shoulder heights
                    hs_patterns[head_idx] = 1
        
        self.chart_patterns['Head_Shoulders'] = hs_patterns
        
    def _detect_triangles(self):
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        triangles = np.zeros(len(self.close))
        
        window = 30
        for i in range(window, len(self.close)):
            recent_highs = []
            recent_lows = []
            
            # Find recent highs and lows
            for j in range(i - window, i):
                if self._is_local_high(j):
                    recent_highs.append((j, self.high[j]))
                if self._is_local_low(j):
                    recent_lows.append((j, self.low[j]))
            
            if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                # Analyze trendlines
                high_slope = self._calculate_trendline_slope(recent_highs[-3:])
                low_slope = self._calculate_trendline_slope(recent_lows[-3:])
                
                # Ascending triangle
                if abs(high_slope) < 0.001 and low_slope > 0.001:
                    triangles[i] = 1
                # Descending triangle
                elif high_slope < -0.001 and abs(low_slope) < 0.001:
                    triangles[i] = -1
                # Symmetrical triangle
                elif high_slope < -0.001 and low_slope > 0.001:
                    triangles[i] = 0.5
        
        self.chart_patterns['Triangles'] = triangles
        
    def _detect_flags_pennants(self):
        """Detect flag and pennant patterns"""
        flags = np.zeros(len(self.close))
        
        window = 15
        for i in range(window * 2, len(self.close)):
            # Look for strong move followed by consolidation
            prev_move = self.close[i - window] - self.close[i - window * 2]
            current_range = np.max(self.high[i - window:i]) - np.min(self.low[i - window:i])
            avg_range = np.mean(self.high[i - window * 2:i - window] - self.low[i - window * 2:i - window])
            
            # Strong initial move
            if abs(prev_move) > avg_range * 3:
                # Followed by consolidation
                if current_range < avg_range * 0.8:
                    flags[i] = np.sign(prev_move)
        
        self.chart_patterns['Flags_Pennants'] = flags
        
    def _detect_channels(self):
        """Detect price channels"""
        channels = np.zeros(len(self.close))
        
        window = 40
        for i in range(window, len(self.close)):
            highs = self.high[i - window:i]
            lows = self.low[i - window:i]
            
            # Calculate channel bounds
            upper_bound = np.percentile(highs, 95)
            lower_bound = np.percentile(lows, 5)
            channel_width = upper_bound - lower_bound
            
            # Check if price is contained within channel
            recent_prices = self.close[i - window//4:i]
            if (np.all(recent_prices <= upper_bound * 1.01) and 
                np.all(recent_prices >= lower_bound * 0.99)):
                channels[i] = channel_width / np.mean(recent_prices)
        
        self.chart_patterns['Channels'] = channels
        
    def _detect_wedges(self):
        """Detect rising and falling wedge patterns"""
        wedges = np.zeros(len(self.close))
        
        window = 25
        for i in range(window, len(self.close)):
            # Find trendlines for highs and lows
            highs = [(j, self.high[j]) for j in range(i - window, i) if self._is_local_high(j)]
            lows = [(j, self.low[j]) for j in range(i - window, i) if self._is_local_low(j)]
            
            if len(highs) >= 3 and len(lows) >= 3:
                high_slope = self._calculate_trendline_slope(highs[-3:])
                low_slope = self._calculate_trendline_slope(lows[-3:])
                
                # Rising wedge (bearish)
                if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
                    wedges[i] = -1
                # Falling wedge (bullish)
                elif high_slope < 0 and low_slope < 0 and high_slope < low_slope:
                    wedges[i] = 1
        
        self.chart_patterns['Wedges'] = wedges
        
    def _discover_cv_patterns(self):
        """Use computer vision to discover novel patterns"""
        
        # Convert price data to image
        price_image = self._price_to_image()
        
        # Apply edge detection
        edges = cv2.Canny(price_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour patterns
        pattern_features = self._analyze_contours(contours)
        
        # Cluster similar patterns
        if len(pattern_features) > 0:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(pattern_features)
            
            kmeans = KMeans(n_clusters=min(10, len(pattern_features)), random_state=42)
            pattern_clusters = kmeans.fit_predict(features_scaled)
            
            self.chart_patterns['CV_Patterns'] = pattern_clusters
        
    def _analyze_support_resistance(self):
        """Comprehensive support and resistance analysis"""
        
        # Find pivot points
        pivots = self._find_pivot_points()
        
        # Cluster pivot points to find levels
        support_levels, resistance_levels = self._cluster_pivot_points(pivots)
        
        # Calculate level strength
        self.support_resistance['Support_Levels'] = self._calculate_level_strength(support_levels, 'support')
        self.support_resistance['Resistance_Levels'] = self._calculate_level_strength(resistance_levels, 'resistance')
        
        # Dynamic levels
        self.support_resistance['Dynamic_Support'] = self._calculate_dynamic_support()
        self.support_resistance['Dynamic_Resistance'] = self._calculate_dynamic_resistance()
        
        # Volume-based levels
        self.support_resistance['Volume_Levels'] = self._calculate_volume_levels()
        
    def _analyze_gaps(self):
        """Comprehensive gap analysis"""
        
        # Detect different types of gaps
        gaps = self._detect_gaps()
        
        # Gap classification
        self.support_resistance['Gap_Analysis'] = self._classify_gaps(gaps)
        
        # Gap fill probability
        self.support_resistance['Gap_Fill_Probability'] = self._calculate_gap_fill_probability(gaps)
        
    def _analyze_volume_price_patterns(self):
        """Volume-price relationship analysis"""
        
        # Volume confirmation patterns
        self.support_resistance['Volume_Confirmation'] = self._volume_confirmation_patterns()
        
        # Volume breakout patterns
        self.support_resistance['Volume_Breakouts'] = self._volume_breakout_patterns()
        
        # Volume exhaustion patterns
        self.support_resistance['Volume_Exhaustion'] = self._volume_exhaustion_patterns()
        
    def _analyze_breakouts(self):
        """Breakout pattern analysis"""
        
        # Breakout detection
        breakouts = self._detect_breakouts()
        
        # Breakout direction prediction
        self.chart_patterns['Breakout_Direction'] = self._predict_breakout_direction(breakouts)
        
        # Breakout strength
        self.chart_patterns['Breakout_Strength'] = self._calculate_breakout_strength(breakouts)
        
        # False breakout detection
        self.chart_patterns['False_Breakouts'] = self._detect_false_breakouts(breakouts)
        
    def _analyze_reversals(self):
        """Reversal pattern analysis"""
        
        # Reversal signals
        reversals = self._detect_reversal_signals()
        
        # Reversal strength
        self.chart_patterns['Reversal_Strength'] = self._calculate_reversal_strength(reversals)
        
        # Reversal confirmation
        self.chart_patterns['Reversal_Confirmation'] = self._reversal_confirmation(reversals)
        
    # Helper methods
    
    def _calculate_pattern_confidence(self):
        """Calculate confidence scores for detected patterns"""
        confidence_scores = {}
        
        for pattern_name, pattern_values in self.candlestick_patterns.items():
            # Calculate historical success rate
            success_rate = self._calculate_historical_success_rate(pattern_values)
            
            # Volume confirmation
            volume_conf = self._volume_pattern_confirmation(pattern_values)
            
            # Context confirmation
            context_conf = self._context_pattern_confirmation(pattern_values)
            
            # Combined confidence
            combined_confidence = (success_rate * 0.5 + volume_conf * 0.3 + context_conf * 0.2)
            
            confidence_scores[pattern_name] = combined_confidence
            
        self.candlestick_patterns['Pattern_Confidence'] = confidence_scores
        
    def _fibonacci_level(self, idx):
        """Calculate proximity to Fibonacci levels"""
        if idx < 50:
            return 0
            
        period_high = np.max(self.high[idx-50:idx])
        period_low = np.min(self.low[idx-50:idx])
        range_size = period_high - period_low
        
        if range_size == 0:
            return 0
            
        current_level = (self.close[idx] - period_low) / range_size
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        # Find closest Fibonacci level
        closest_fib = min(fib_levels, key=lambda x: abs(x - current_level))
        proximity = 1 - abs(current_level - closest_fib)
        
        return proximity
        
    def _sr_proximity(self, idx):
        """Calculate proximity to support/resistance levels"""
        if idx < 20:
            return 0
            
        # Simple S/R calculation
        recent_highs = np.max(self.high[idx-20:idx])
        recent_lows = np.min(self.low[idx-20:idx])
        
        high_proximity = 1 - abs(self.close[idx] - recent_highs) / recent_highs
        low_proximity = 1 - abs(self.close[idx] - recent_lows) / recent_lows
        
        return max(high_proximity, low_proximity)
        
    def _analyze_clusters(self, kmeans_clusters, dbscan_clusters):
        """Analyze cluster characteristics"""
        cluster_analysis = {}
        
        # Analyze K-means clusters
        for cluster_id in set(kmeans_clusters):
            cluster_indices = np.where(kmeans_clusters == cluster_id)[0]
            
            if len(cluster_indices) > 5:  # Minimum cluster size
                # Calculate cluster characteristics
                cluster_returns = []
                for idx in cluster_indices:
                    if idx < len(self.close) - 5:
                        future_return = (self.close[idx + 5] - self.close[idx]) / self.close[idx]
                        cluster_returns.append(future_return)
                
                if cluster_returns:
                    cluster_analysis[f'KMeans_Cluster_{cluster_id}'] = {
                        'avg_return': np.mean(cluster_returns),
                        'success_rate': np.mean(np.array(cluster_returns) > 0),
                        'volatility': np.std(cluster_returns),
                        'sample_size': len(cluster_returns)
                    }
        
        return cluster_analysis
        
    def _cluster_signals(self, clusters, features):
        """Generate trading signals based on cluster analysis"""
        signals = np.zeros(len(self.close))
        
        # Analyze each cluster's performance
        cluster_performance = {}
        for cluster_id in set(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) > 10:
                returns = []
                for idx in cluster_indices:
                    if idx < len(self.close) - 1:
                        ret = (self.close[idx + 1] - self.close[idx]) / self.close[idx]
                        returns.append(ret)
                
                if returns:
                    avg_return = np.mean(returns)
                    success_rate = np.mean(np.array(returns) > 0)
                    cluster_performance[cluster_id] = (avg_return, success_rate)
        
        # Generate signals based on cluster performance
        for i, cluster_id in enumerate(clusters):
            if cluster_id in cluster_performance:
                avg_return, success_rate = cluster_performance[cluster_id]
                if success_rate > 0.6 and avg_return > 0.001:
                    signals[i] = 1  # Bullish signal
                elif success_rate < 0.4 and avg_return < -0.001:
                    signals[i] = -1  # Bearish signal
        
        return signals
        
    def _analyze_anomalies(self, anomalies, features):
        """Analyze characteristics of anomalous patterns"""
        anomaly_analysis = {}
        
        anomaly_indices = np.where(anomalies == -1)[0]
        normal_indices = np.where(anomalies == 1)[0]
        
        if len(anomaly_indices) > 0 and len(normal_indices) > 0:
            # Compare anomalous vs normal patterns
            anomaly_features = features[anomaly_indices]
            normal_features = features[normal_indices]
            
            # Statistical comparison
            for i in range(features.shape[1]):
                anomaly_mean = np.mean(anomaly_features[:, i])
                normal_mean = np.mean(normal_features[:, i])
                
                if abs(anomaly_mean - normal_mean) > np.std(normal_features[:, i]):
                    anomaly_analysis[f'Feature_{i}_Difference'] = anomaly_mean - normal_mean
        
        return anomaly_analysis
        
    def _create_candlestick_sequences(self):
        """Create sequences for pattern mining"""
        sequences = []
        
        for i in range(len(self.close)):
            # Categorize candlestick
            body_size = abs(self.close[i] - self.open[i])
            range_size = self.high[i] - self.low[i]
            
            if range_size == 0:
                candle_type = 'doji'
            elif body_size / range_size > 0.7:
                candle_type = 'large_body'
            elif body_size / range_size < 0.3:
                candle_type = 'small_body'
            else:
                candle_type = 'medium_body'
            
            # Add direction
            if self.close[i] > self.open[i]:
                candle_type += '_bull'
            else:
                candle_type += '_bear'
            
            sequences.append(candle_type)
        
        return sequences
        
    def _find_frequent_sequences(self, sequences):
        """Find frequent candlestick sequences"""
        sequence_patterns = {}
        sequence_length = 3
        
        for i in range(len(sequences) - sequence_length + 1):
            pattern = tuple(sequences[i:i + sequence_length])
            
            if pattern not in sequence_patterns:
                sequence_patterns[pattern] = []
            
            sequence_patterns[pattern].append(i)
        
        # Filter frequent patterns
        frequent_patterns = {k: v for k, v in sequence_patterns.items() if len(v) >= 5}
        
        return frequent_patterns
        
    def _analyze_sequence_predictions(self, sequences):
        """Analyze predictive power of sequences"""
        predictions = {}
        
        frequent_patterns = self._find_frequent_sequences(sequences)
        
        for pattern, occurrences in frequent_patterns.items():
            future_moves = []
            
            for occurrence in occurrences:
                if occurrence + len(pattern) < len(self.close):
                    future_idx = occurrence + len(pattern)
                    future_return = (self.close[future_idx] - self.close[occurrence + len(pattern) - 1]) / self.close[occurrence + len(pattern) - 1]
                    future_moves.append(future_return)
            
            if future_moves:
                predictions[pattern] = {
                    'avg_return': np.mean(future_moves),
                    'success_rate': np.mean(np.array(future_moves) > 0),
                    'sample_size': len(future_moves)
                }
        
        return predictions
        
    def _find_local_peak(self, start, end):
        """Find local peak in given range"""
        if start >= end or end >= len(self.high):
            return None
            
        max_idx = np.argmax(self.high[start:end]) + start
        max_val = self.high[max_idx]
        
        return (max_idx, max_val)
        
    def _is_local_high(self, idx, window=3):
        """Check if index is a local high"""
        if idx < window or idx >= len(self.high) - window:
            return False
            
        return self.high[idx] == np.max(self.high[idx-window:idx+window+1])
        
    def _is_local_low(self, idx, window=3):
        """Check if index is a local low"""
        if idx < window or idx >= len(self.low) - window:
            return False
            
        return self.low[idx] == np.min(self.low[idx-window:idx+window+1])
        
    def _calculate_trendline_slope(self, points):
        """Calculate slope of trendline through points"""
        if len(points) < 2:
            return 0
            
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # Linear regression
        slope, _, _, _, _ = stats.linregress(x_vals, y_vals)
        return slope
        
    def _price_to_image(self, width=100, height=50):
        """Convert price data to image for CV analysis"""
        # Normalize price data
        normalized_prices = (self.close - np.min(self.close)) / (np.max(self.close) - np.min(self.close))
        
        # Create image
        image = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(min(width, len(normalized_prices))):
            y_pos = int((1 - normalized_prices[i]) * (height - 1))
            image[y_pos, i] = 255
        
        return image
        
    def _analyze_contours(self, contours):
        """Analyze contour features for pattern recognition"""
        features = []
        
        for contour in contours:
            if len(contour) > 5:
                # Basic contour features
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Shape features
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                features.append([area, perimeter, solidity, aspect_ratio])
        
        return features
        
    def _find_pivot_points(self):
        """Find pivot points in price data"""
        pivots = {'highs': [], 'lows': []}
        
        for i in range(5, len(self.close) - 5):
            if self._is_local_high(i, 5):
                pivots['highs'].append((i, self.high[i]))
            if self._is_local_low(i, 5):
                pivots['lows'].append((i, self.low[i]))
        
        return pivots
        
    def _cluster_pivot_points(self, pivots):
        """Cluster pivot points to find support/resistance levels"""
        
        high_prices = [p[1] for p in pivots['highs']]
        low_prices = [p[1] for p in pivots['lows']]
        
        resistance_levels = []
        support_levels = []
        
        if len(high_prices) > 2:
            # Cluster resistance levels
            high_prices_array = np.array(high_prices).reshape(-1, 1)
            kmeans_high = KMeans(n_clusters=min(5, len(high_prices)), random_state=42)
            high_clusters = kmeans_high.fit_predict(high_prices_array)
            resistance_levels = kmeans_high.cluster_centers_.flatten()
        
        if len(low_prices) > 2:
            # Cluster support levels
            low_prices_array = np.array(low_prices).reshape(-1, 1)
            kmeans_low = KMeans(n_clusters=min(5, len(low_prices)), random_state=42)
            low_clusters = kmeans_low.fit_predict(low_prices_array)
            support_levels = kmeans_low.cluster_centers_.flatten()
        
        return support_levels, resistance_levels
        
    def _calculate_level_strength(self, levels, level_type):
        """Calculate strength of support/resistance levels"""
        level_strength = {}
        
        for level in levels:
            touches = 0
            bounces = 0
            
            for i in range(len(self.close)):
                # Check if price touched the level
                tolerance = level * 0.01  # 1% tolerance
                
                if level_type == 'support':
                    if abs(self.low[i] - level) <= tolerance:
                        touches += 1
                        # Check for bounce
                        if i < len(self.close) - 1 and self.close[i + 1] > self.close[i]:
                            bounces += 1
                else:  # resistance
                    if abs(self.high[i] - level) <= tolerance:
                        touches += 1
                        # Check for rejection
                        if i < len(self.close) - 1 and self.close[i + 1] < self.close[i]:
                            bounces += 1
            
            strength = bounces / touches if touches > 0 else 0
            level_strength[level] = {
                'touches': touches,
                'bounces': bounces,
                'strength': strength
            }
        
        return level_strength
        
    def _calculate_dynamic_support(self):
        """Calculate dynamic support levels"""
        dynamic_support = np.zeros(len(self.close))
        
        for i in range(20, len(self.close)):
            # Use moving averages as dynamic support
            sma20 = np.mean(self.close[i-20:i])
            sma50 = np.mean(self.close[i-min(50, i):i])
            
            # Check if price is above MAs
            if self.close[i] > sma20 and self.close[i] > sma50:
                dynamic_support[i] = max(sma20, sma50)
        
        return dynamic_support
        
    def _calculate_dynamic_resistance(self):
        """Calculate dynamic resistance levels"""
        dynamic_resistance = np.zeros(len(self.close))
        
        for i in range(20, len(self.close)):
            # Use recent highs as dynamic resistance
            recent_high = np.max(self.high[i-20:i])
            
            if self.close[i] < recent_high * 0.95:
                dynamic_resistance[i] = recent_high
        
        return dynamic_resistance
        
    def _calculate_volume_levels(self):
        """Calculate volume-based support/resistance levels"""
        volume_levels = {}
        
        # Volume-weighted price levels
        price_volume_pairs = list(zip(self.close, self.volume))
        
        # Group by price levels
        price_ranges = {}
        for price, volume in price_volume_pairs:
            price_bucket = round(price / (np.max(self.close) * 0.01)) * (np.max(self.close) * 0.01)
            
            if price_bucket not in price_ranges:
                price_ranges[price_bucket] = 0
            price_ranges[price_bucket] += volume
        
        # Find high volume levels
        sorted_levels = sorted(price_ranges.items(), key=lambda x: x[1], reverse=True)
        top_volume_levels = sorted_levels[:5]  # Top 5 volume levels
        
        volume_levels = {level: volume for level, volume in top_volume_levels}
        
        return volume_levels
        
    def _detect_gaps(self):
        """Detect price gaps"""
        gaps = []
        
        for i in range(1, len(self.close)):
            # Gap up
            if self.low[i] > self.high[i-1]:
                gap_size = self.low[i] - self.high[i-1]
                gaps.append({
                    'index': i,
                    'type': 'gap_up',
                    'size': gap_size,
                    'size_percent': gap_size / self.close[i-1] * 100
                })
            
            # Gap down
            elif self.high[i] < self.low[i-1]:
                gap_size = self.low[i-1] - self.high[i]
                gaps.append({
                    'index': i,
                    'type': 'gap_down',
                    'size': gap_size,
                    'size_percent': gap_size / self.close[i-1] * 100
                })
        
        return gaps
        
    def _classify_gaps(self, gaps):
        """Classify gaps by type and significance"""
        classified_gaps = {}
        
        for gap in gaps:
            gap_type = 'common'  # Default
            
            if gap['size_percent'] > 3:
                gap_type = 'significant'
            elif gap['size_percent'] > 1:
                gap_type = 'moderate'
            
            # Check for breakaway gaps (high volume)
            idx = gap['index']
            if idx < len(self.volume):
                avg_volume = np.mean(self.volume[max(0, idx-10):idx])
                if self.volume[idx] > avg_volume * 1.5:
                    gap_type = 'breakaway'
            
            classified_gaps[idx] = {
                'classification': gap_type,
                'direction': gap['type'],
                'size_percent': gap['size_percent']
            }
        
        return classified_gaps
        
    def _calculate_gap_fill_probability(self, gaps):
        """Calculate probability of gap filling"""
        fill_probabilities = {}
        
        for gap in gaps:
            idx = gap['index']
            gap_level = self.high[idx-1] if gap['type'] == 'gap_up' else self.low[idx-1]
            
            # Look for gap fill in subsequent periods
            filled = False
            fill_time = None
            
            for j in range(idx + 1, min(idx + 20, len(self.close))):
                if gap['type'] == 'gap_up' and self.low[j] <= gap_level:
                    filled = True
                    fill_time = j - idx
                    break
                elif gap['type'] == 'gap_down' and self.high[j] >= gap_level:
                    filled = True
                    fill_time = j - idx
                    break
            
            # Calculate probability based on gap characteristics
            base_prob = 0.7  # Base probability
            
            # Adjust based on gap size
            if gap['size_percent'] > 2:
                base_prob *= 0.8  # Large gaps less likely to fill quickly
            
            # Adjust based on volume
            if idx < len(self.volume):
                avg_volume = np.mean(self.volume[max(0, idx-10):idx])
                if self.volume[idx] > avg_volume * 2:
                    base_prob *= 0.6  # High volume gaps less likely to fill
            
            fill_probabilities[idx] = {
                'probability': base_prob,
                'filled': filled,
                'fill_time': fill_time
            }
        
        return fill_probabilities
    
    # Additional helper methods for volume and breakout analysis
    
    def _volume_confirmation_patterns(self):
        """Volume confirmation for price patterns"""
        confirmations = np.zeros(len(self.close))
        
        for i in range(20, len(self.close)):
            # Calculate average volume
            avg_volume = np.mean(self.volume[i-20:i])
            
            # Price movement
            price_change = (self.close[i] - self.close[i-1]) / self.close[i-1]
            
            # Volume confirmation
            if abs(price_change) > 0.02:  # Significant move
                if self.volume[i] > avg_volume * 1.5:  # High volume
                    confirmations[i] = np.sign(price_change)
        
        return confirmations
    
    def _volume_breakout_patterns(self):
        """Volume-based breakout patterns"""
        breakouts = np.zeros(len(self.close))
        
        for i in range(20, len(self.close)):
            # Recent high/low
            recent_high = np.max(self.high[i-20:i])
            recent_low = np.min(self.low[i-20:i])
            
            # Volume spike
            avg_volume = np.mean(self.volume[i-20:i])
            volume_spike = self.volume[i] > avg_volume * 2
            
            # Breakout detection
            if volume_spike:
                if self.high[i] > recent_high:
                    breakouts[i] = 1  # Bullish breakout
                elif self.low[i] < recent_low:
                    breakouts[i] = -1  # Bearish breakout
        
        return breakouts
    
    def _volume_exhaustion_patterns(self):
        """Volume exhaustion patterns"""
        exhaustion = np.zeros(len(self.close))
        
        for i in range(10, len(self.close)):
            # Trend direction
            trend = np.sign(self.close[i] - self.close[i-10])
            
            # Volume analysis
            recent_volumes = self.volume[i-5:i+1]
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            
            # Exhaustion: strong trend with declining volume
            if abs(trend) > 0 and volume_trend < 0:
                exhaustion[i] = -trend  # Opposite signal
        
        return exhaustion
    
    def _detect_breakouts(self):
        """Detect breakout patterns"""
        breakouts = np.zeros(len(self.close))
        
        window = 20
        for i in range(window, len(self.close)):
            # Consolidation period
            recent_high = np.max(self.high[i-window:i])
            recent_low = np.min(self.low[i-window:i])
            consolidation_range = recent_high - recent_low
            
            # Breakout threshold
            threshold = consolidation_range * 0.1
            
            # Detect breakout
            if self.high[i] > recent_high + threshold:
                breakouts[i] = 1  # Upward breakout
            elif self.low[i] < recent_low - threshold:
                breakouts[i] = -1  # Downward breakout
        
        return breakouts
    
    def _predict_breakout_direction(self, breakouts):
        """Predict breakout direction"""
        direction_prediction = np.zeros(len(self.close))
        
        window = 30
        for i in range(window, len(self.close)):
            if breakouts[i] == 0:  # No current breakout
                # Analyze momentum indicators for direction
                momentum = (self.close[i] - self.close[i-10]) / self.close[i-10]
                volume_trend = np.polyfit(range(10), self.volume[i-9:i+1], 1)[0]
                
                # Combine signals
                if momentum > 0.02 and volume_trend > 0:
                    direction_prediction[i] = 1  # Likely upward breakout
                elif momentum < -0.02 and volume_trend > 0:
                    direction_prediction[i] = -1  # Likely downward breakout
        
        return direction_prediction
    
    def _calculate_breakout_strength(self, breakouts):
        """Calculate breakout strength"""
        strength = np.zeros(len(self.close))
        
        for i in range(len(self.close)):
            if breakouts[i] != 0:
                # Volume confirmation
                avg_volume = np.mean(self.volume[max(0, i-20):i])
                volume_strength = self.volume[i] / avg_volume if avg_volume > 0 else 1
                
                # Price movement strength
                price_movement = abs(self.close[i] - self.open[i]) / self.open[i]
                
                # Combined strength
                strength[i] = (volume_strength + price_movement * 100) / 2
        
        return strength
    
    def _detect_false_breakouts(self, breakouts):
        """Detect false breakout patterns"""
        false_breakouts = np.zeros(len(self.close))
        
        for i in range(len(self.close) - 5):
            if breakouts[i] != 0:
                # Check if breakout failed within next 5 periods
                failed = False
                
                if breakouts[i] == 1:  # Upward breakout
                    # Check if price fell back
                    for j in range(i + 1, min(i + 6, len(self.close))):
                        if self.close[j] < self.close[i] * 0.98:
                            failed = True
                            break
                else:  # Downward breakout
                    # Check if price recovered
                    for j in range(i + 1, min(i + 6, len(self.close))):
                        if self.close[j] > self.close[i] * 1.02:
                            failed = True
                            break
                
                if failed:
                    false_breakouts[i] = 1
        
        return false_breakouts
    
    def _detect_reversal_signals(self):
        """Detect reversal signals"""
        reversals = np.zeros(len(self.close))
        
        window = 10
        for i in range(window, len(self.close)):
            # Trend analysis
            trend = (self.close[i] - self.close[i-window]) / self.close[i-window]
            
            # Momentum divergence
            price_momentum = (self.close[i] - self.close[i-5]) / self.close[i-5]
            volume_momentum = (self.volume[i] - np.mean(self.volume[i-5:i])) / np.mean(self.volume[i-5:i])
            
            # Reversal conditions
            if trend > 0.05 and price_momentum < 0 and volume_momentum < -0.2:
                reversals[i] = -1  # Bearish reversal
            elif trend < -0.05 and price_momentum > 0 and volume_momentum < -0.2:
                reversals[i] = 1  # Bullish reversal
        
        return reversals
    
    def _calculate_reversal_strength(self, reversals):
        """Calculate reversal signal strength"""
        strength = np.zeros(len(self.close))
        
        for i in range(len(self.close)):
            if reversals[i] != 0:
                # Multiple confirmation factors
                factors = []
                
                # Volume factor
                avg_volume = np.mean(self.volume[max(0, i-10):i])
                volume_factor = self.volume[i] / avg_volume if avg_volume > 0 else 1
                factors.append(min(volume_factor, 3))  # Cap at 3x
                
                # Price action factor
                price_range = self.high[i] - self.low[i]
                avg_range = np.mean([self.high[j] - self.low[j] for j in range(max(0, i-10), i)])
                range_factor = price_range / avg_range if avg_range > 0 else 1
                factors.append(min(range_factor, 2))  # Cap at 2x
                
                # Trend change factor
                short_trend = (self.close[i] - self.close[i-5]) / self.close[i-5] if i >= 5 else 0
                long_trend = (self.close[i] - self.close[i-10]) / self.close[i-10] if i >= 10 else 0
                trend_factor = abs(short_trend - long_trend) * 100
                factors.append(min(trend_factor, 2))
                
                strength[i] = np.mean(factors)
        
        return strength
    
    def _reversal_confirmation(self, reversals):
        """Reversal confirmation analysis"""
        confirmation = np.zeros(len(self.close))
        
        for i in range(len(self.close) - 3):
            if reversals[i] != 0:
                # Look for confirmation in next periods
                confirmed = False
                
                if reversals[i] == 1:  # Bullish reversal
                    # Check for upward movement
                    for j in range(i + 1, min(i + 4, len(self.close))):
                        if self.close[j] > self.close[i] * 1.01:
                            confirmed = True
                            break
                else:  # Bearish reversal
                    # Check for downward movement
                    for j in range(i + 1, min(i + 4, len(self.close))):
                        if self.close[j] < self.close[i] * 0.99:
                            confirmed = True
                            break
                
                if confirmed:
                    confirmation[i] = 1
        
        return confirmation
    
    def _calculate_historical_success_rate(self, pattern_values):
        """Calculate historical success rate of patterns"""
        successes = 0
        total_patterns = 0
        
        for i in range(len(pattern_values) - 5):
            if pattern_values[i] != 0:
                total_patterns += 1
                
                # Check if pattern was successful (price moved in predicted direction)
                pattern_direction = np.sign(pattern_values[i])
                future_move = (self.close[i + 5] - self.close[i]) / self.close[i]
                
                if pattern_direction * future_move > 0:
                    successes += 1
        
        return successes / total_patterns if total_patterns > 0 else 0
    
    def _volume_pattern_confirmation(self, pattern_values):
        """Volume confirmation for patterns"""
        confirmations = 0
        total_patterns = 0
        
        for i in range(len(pattern_values)):
            if pattern_values[i] != 0:
                total_patterns += 1
                
                # Check volume confirmation
                avg_volume = np.mean(self.volume[max(0, i-10):i+1])
                if self.volume[i] > avg_volume * 1.2:
                    confirmations += 1
        
        return confirmations / total_patterns if total_patterns > 0 else 0
    
    def _context_pattern_confirmation(self, pattern_values):
        """Context-based pattern confirmation"""
        confirmations = 0
        total_patterns = 0
        
        for i in range(20, len(pattern_values)):
            if pattern_values[i] != 0:
                total_patterns += 1
                
                # Trend context
                trend = (self.close[i] - self.close[i-20]) / self.close[i-20]
                pattern_direction = np.sign(pattern_values[i])
                
                # Pattern aligned with trend gets bonus
                if pattern_direction * trend > 0:
                    confirmations += 1
        
        return confirmations / total_patterns if total_patterns > 0 else 0
    
    def get_current_patterns(self):
        """Get current active patterns"""
        current_patterns = {}
        latest_idx = -1
        
        # Candlestick patterns
        for pattern, values in self.candlestick_patterns.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                if not pd.isna(values[latest_idx]) and values[latest_idx] != 0:
                    current_patterns[f'Candlestick_{pattern}'] = float(values[latest_idx])
        
        # Chart patterns
        for pattern, values in self.chart_patterns.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                if not pd.isna(values[latest_idx]) and values[latest_idx] != 0:
                    current_patterns[f'Chart_{pattern}'] = float(values[latest_idx])
        
        # ML patterns
        for pattern, values in self.ml_patterns.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                if not pd.isna(values[latest_idx]) and values[latest_idx] != 0:
                    current_patterns[f'ML_{pattern}'] = float(values[latest_idx])
        
        return current_patterns
    
    def get_support_resistance_levels(self):
        """Get current support and resistance levels"""
        return self.support_resistance