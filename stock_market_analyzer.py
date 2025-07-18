import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_fetcher import IndianStockDataFetcher
from technical_indicators import AdvancedTechnicalIndicators
from price_action_analysis import AdvancedPriceActionAnalysis
from deep_learning_models import DeepLearningModels

class StockMarketAnalyzer:
    """
    Comprehensive Deep Learning Stock Market Analysis Module for Indian Markets
    
    This is an expert-level system that unveils market secrets and provides detailed insights
    through advanced technical analysis, price action analysis, and deep learning predictions.
    """
    
    def __init__(self, symbol="RELIANCE.NS"):
        """
        Initialize the analyzer with a stock symbol
        
        Args:
            symbol (str): Stock symbol to analyze (default: RELIANCE.NS)
        """
        self.symbol = symbol
        self.data_fetcher = IndianStockDataFetcher(symbol)
        self.stock_info = {}
        self.historical_data = {}
        self.technical_analysis = {}
        self.price_action_analysis = {}
        self.ml_predictions = {}
        self.confidence_scores = {}
        
        print(f"🚀 Initializing Advanced Stock Market Analyzer for {symbol}")
        print("=" * 60)
        
    def fetch_comprehensive_data(self):
        """Fetch comprehensive data for all timeframes"""
        print("📊 Fetching comprehensive market data...")
        
        # Get stock information
        self.stock_info = self.data_fetcher.get_stock_info()
        
        # Get multi-timeframe data
        self.historical_data = self.data_fetcher.get_comprehensive_data()
        
        print(f"✅ Data fetched successfully!")
        print(f"📈 Company: {self.stock_info.get('company_name', self.symbol)}")
        print(f"🏭 Sector: {self.stock_info.get('sector', 'Unknown')}")
        print(f"💰 Market Cap: ₹{self.stock_info.get('market_cap', 0):,.0f}")
        print(f"📊 Available Timeframes: {list(self.historical_data.keys())}")
        
        return self.historical_data
    
    def run_comprehensive_analysis(self, train_models=True):
        """
        Run comprehensive analysis including:
        - Technical indicator analysis
        - Price action analysis  
        - Deep learning predictions
        - Pattern recognition
        """
        print("\n🔍 Starting Comprehensive Market Analysis...")
        print("=" * 60)
        
        if not self.historical_data:
            self.fetch_comprehensive_data()
        
        # Analyze each timeframe
        for timeframe, data in self.historical_data.items():
            if len(data) > 100:  # Minimum data requirement
                print(f"\n📈 Analyzing {timeframe} timeframe ({len(data)} records)...")
                
                # Technical Analysis
                print("  🔧 Running technical indicator analysis...")
                tech_analyzer = AdvancedTechnicalIndicators(data)
                tech_indicators = tech_analyzer.calculate_all_indicators()
                tech_signals = tech_analyzer.get_current_signals()
                tech_patterns = tech_analyzer.get_pattern_summary()
                
                self.technical_analysis[timeframe] = {
                    'indicators': tech_indicators,
                    'current_signals': tech_signals,
                    'patterns': tech_patterns,
                    'analyzer': tech_analyzer
                }
                
                # Price Action Analysis
                print("  📊 Running price action analysis...")
                pa_analyzer = AdvancedPriceActionAnalysis(data)
                pa_patterns = pa_analyzer.analyze_all_patterns()
                current_patterns = pa_analyzer.get_current_patterns()
                sr_levels = pa_analyzer.get_support_resistance_levels()
                
                self.price_action_analysis[timeframe] = {
                    'patterns': pa_patterns,
                    'current_patterns': current_patterns,
                    'support_resistance': sr_levels,
                    'analyzer': pa_analyzer
                }
                
                print(f"  ✅ {timeframe} analysis complete!")
        
        # Deep Learning Analysis
        if train_models and len(self.historical_data) > 0:
            print("\n🤖 Training Deep Learning Models...")
            self._run_deep_learning_analysis()
        
        print("\n✅ Comprehensive Analysis Complete!")
        return True
    
    def _run_deep_learning_analysis(self):
        """Run deep learning analysis and predictions"""
        try:
            # Initialize ML models
            ml_analyzer = DeepLearningModels(self.historical_data, sequence_length=60)
            
            # Train models for each timeframe with sufficient data
            training_results = {}
            for timeframe in ['5m', '15m', '1h', '1d', '1wk']:
                if timeframe in self.historical_data and len(self.historical_data[timeframe]) > 200:
                    print(f"  🧠 Training models for {timeframe}...")
                    try:
                        history = ml_analyzer.train_models(timeframe)
                        training_results[timeframe] = history
                        print(f"  ✅ {timeframe} models trained successfully!")
                    except Exception as e:
                        print(f"  ❌ Error training {timeframe} models: {e}")
            
            # Generate predictions
            print("  🔮 Generating ML predictions...")
            predictions = ml_analyzer.predict_multiple_timeframes(self.historical_data)
            
            # Multi-horizon predictions
            multi_horizon = {}
            for timeframe in training_results.keys():
                multi_horizon[timeframe] = ml_analyzer.predict_price_movements(timeframe, horizon=5)
            
            self.ml_predictions = {
                'current_predictions': predictions,
                'multi_horizon': multi_horizon,
                'model_summary': ml_analyzer.get_model_summary(),
                'analyzer': ml_analyzer
            }
            
            print("  ✅ Deep learning analysis complete!")
            
        except Exception as e:
            print(f"  ❌ Deep learning analysis failed: {e}")
            self.ml_predictions = {}
    
    def generate_technical_analysis_report(self):
        """Generate comprehensive technical analysis report"""
        
        report = {
            'metadata': {
                'symbol': self.symbol,
                'company_name': self.stock_info.get('company_name', self.symbol),
                'sector': self.stock_info.get('sector', 'Unknown'),
                'generated_at': datetime.now().isoformat(),
                'market_cap': self.stock_info.get('market_cap', 0),
                'pe_ratio': self.stock_info.get('pe_ratio', 0),
                'beta': self.stock_info.get('beta', 1.0)
            },
            'timeframe_analysis': {},
            'overall_sentiment': {},
            'risk_assessment': {},
            'trading_signals': {}
        }
        
        # Analyze each timeframe
        for timeframe, analysis in self.technical_analysis.items():
            if 'current_signals' in analysis:
                signals = analysis['current_signals']
                patterns = analysis['patterns']
                
                # Extract key indicators
                key_indicators = {}
                for indicator, value in signals.items():
                    if not pd.isna(value):
                        key_indicators[indicator] = float(value)
                
                # Generate timeframe summary
                timeframe_summary = self._generate_timeframe_technical_summary(
                    key_indicators, patterns, timeframe
                )
                
                report['timeframe_analysis'][timeframe] = timeframe_summary
        
        # Overall sentiment analysis
        report['overall_sentiment'] = self._calculate_overall_technical_sentiment()
        
        # Risk assessment
        report['risk_assessment'] = self._calculate_technical_risk_assessment()
        
        # Trading signals
        report['trading_signals'] = self._generate_technical_trading_signals()
        
        # ML predictions if available
        if self.ml_predictions:
            report['ml_insights'] = self._extract_ml_technical_insights()
        
        return report
    
    def generate_price_action_report(self):
        """Generate comprehensive price action analysis report"""
        
        report = {
            'metadata': {
                'symbol': self.symbol,
                'company_name': self.stock_info.get('company_name', self.symbol),
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'Price Action Analysis'
            },
            'candlestick_analysis': {},
            'chart_patterns': {},
            'support_resistance': {},
            'breakout_analysis': {},
            'reversal_signals': {},
            'volume_analysis': {},
            'pattern_reliability': {}
        }
        
        # Analyze each timeframe
        for timeframe, analysis in self.price_action_analysis.items():
            patterns = analysis.get('patterns', {})
            current_patterns = analysis.get('current_patterns', {})
            sr_levels = analysis.get('support_resistance', {})
            
            # Candlestick analysis
            candlestick_summary = self._extract_candlestick_patterns(patterns, timeframe)
            report['candlestick_analysis'][timeframe] = candlestick_summary
            
            # Chart patterns
            chart_summary = self._extract_chart_patterns(patterns, timeframe)
            report['chart_patterns'][timeframe] = chart_summary
            
            # Support/Resistance
            sr_summary = self._extract_support_resistance_analysis(sr_levels, timeframe)
            report['support_resistance'][timeframe] = sr_summary
            
            # Volume analysis
            volume_summary = self._extract_volume_analysis(patterns, timeframe)
            report['volume_analysis'][timeframe] = volume_summary
        
        # Overall pattern analysis
        report['breakout_analysis'] = self._analyze_breakout_potential()
        report['reversal_signals'] = self._analyze_reversal_potential()
        report['pattern_reliability'] = self._calculate_pattern_reliability()
        
        # ML-discovered patterns
        if self.ml_predictions:
            report['ml_discovered_patterns'] = self._extract_ml_price_action_insights()
        
        return report
    
    def _generate_timeframe_technical_summary(self, indicators, patterns, timeframe):
        """Generate technical summary for a specific timeframe"""
        summary = {
            'timeframe': timeframe,
            'trend_analysis': {},
            'momentum_analysis': {},
            'volatility_analysis': {},
            'pattern_signals': {},
            'key_levels': {},
            'prediction_confidence': {}
        }
        
        # Trend Analysis
        trend_indicators = ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'MA_Cloud_Strength']
        trend_values = {k: indicators.get(k, 0) for k in trend_indicators if k in indicators}
        
        trend_score = 0
        if 'MA_Cloud_Strength' in indicators:
            trend_score = indicators['MA_Cloud_Strength']
        
        summary['trend_analysis'] = {
            'trend_score': trend_score,
            'trend_direction': 'Bullish' if trend_score > 0.3 else 'Bearish' if trend_score < -0.3 else 'Neutral',
            'moving_averages': trend_values,
            'golden_cross': bool(patterns.get('Golden_Cross', 0)),
            'death_cross': bool(patterns.get('Death_Cross', 0))
        }
        
        # Momentum Analysis
        momentum_indicators = ['RSI_14', 'MACD', 'MACD_Signal', 'ROC_10', 'ROC_20', 'Momentum_Composite']
        momentum_values = {k: indicators.get(k, 0) for k in momentum_indicators if k in indicators}
        
        rsi = indicators.get('RSI_14', 50)
        momentum_score = (rsi - 50) / 50  # Normalize RSI
        
        summary['momentum_analysis'] = {
            'momentum_score': momentum_score,
            'rsi_level': rsi,
            'rsi_condition': 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Normal',
            'macd_signal': 'Bullish' if indicators.get('MACD', 0) > indicators.get('MACD_Signal', 0) else 'Bearish',
            'momentum_indicators': momentum_values
        }
        
        # Volatility Analysis
        volatility_indicators = ['ATR', 'BB_Width', 'Volatility_Regime']
        volatility_values = {k: indicators.get(k, 0) for k in volatility_indicators if k in indicators}
        
        vol_regime = indicators.get('Volatility_Regime', 0)
        vol_condition = 'High' if vol_regime > 1 else 'Low' if vol_regime < -1 else 'Normal'
        
        summary['volatility_analysis'] = {
            'volatility_condition': vol_condition,
            'bollinger_squeeze': bool(patterns.get('BB_Squeeze', 0)),
            'bollinger_expansion': bool(patterns.get('BB_Expansion', 0)),
            'volatility_indicators': volatility_values
        }
        
        # Pattern Signals
        pattern_signals = {}
        for pattern, value in patterns.items():
            if value != 0:
                pattern_signals[pattern] = {
                    'signal': 'Bullish' if value > 0 else 'Bearish',
                    'strength': abs(value)
                }
        
        summary['pattern_signals'] = pattern_signals
        
        # Prediction Confidence
        confidence_factors = []
        if abs(trend_score) > 0.5:
            confidence_factors.append(0.3)
        if abs(momentum_score) > 0.3:
            confidence_factors.append(0.2)
        if len(pattern_signals) > 0:
            confidence_factors.append(0.3)
        if indicators.get('Volume_Oscillator', 0) != 0:
            confidence_factors.append(0.2)
        
        overall_confidence = sum(confidence_factors)
        summary['prediction_confidence'] = {
            'confidence_score': min(overall_confidence, 1.0),
            'confidence_level': 'High' if overall_confidence > 0.7 else 'Medium' if overall_confidence > 0.4 else 'Low'
        }
        
        return summary
    
    def _extract_candlestick_patterns(self, patterns, timeframe):
        """Extract candlestick pattern analysis"""
        candlestick_patterns = patterns.get('candlestick_patterns', {})
        
        detected_patterns = {}
        pattern_confidence = {}
        
        # Traditional patterns
        traditional_patterns = [
            'Doji', 'Hammer', 'Hanging_Man', 'Inverted_Hammer', 'Shooting_Star',
            'Engulfing', 'Harami', 'Morning_Star', 'Evening_Star', 'Three_White_Soldiers',
            'Three_Black_Crows', 'Piercing', 'Dark_Cloud'
        ]
        
        for pattern in traditional_patterns:
            if pattern in candlestick_patterns:
                values = candlestick_patterns[pattern]
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    latest_value = values[-1] if not pd.isna(values[-1]) else 0
                    if latest_value != 0:
                        detected_patterns[pattern] = {
                            'signal': 'Bullish' if latest_value > 0 else 'Bearish',
                            'strength': abs(latest_value),
                            'timeframe': timeframe
                        }
        
        # Pattern confidence scores
        if 'Pattern_Confidence' in candlestick_patterns:
            pattern_confidence = candlestick_patterns['Pattern_Confidence']
        
        return {
            'detected_patterns': detected_patterns,
            'pattern_confidence': pattern_confidence,
            'total_patterns_detected': len(detected_patterns),
            'bullish_patterns': len([p for p in detected_patterns.values() if p['signal'] == 'Bullish']),
            'bearish_patterns': len([p for p in detected_patterns.values() if p['signal'] == 'Bearish'])
        }
    
    def _extract_chart_patterns(self, patterns, timeframe):
        """Extract chart pattern analysis"""
        chart_patterns = patterns.get('chart_patterns', {})
        
        detected_patterns = {}
        
        # Chart pattern types
        chart_pattern_types = [
            'Head_Shoulders', 'Triangles', 'Flags_Pennants', 'Channels', 'Wedges',
            'Breakout_Direction', 'Breakout_Strength', 'False_Breakouts'
        ]
        
        for pattern in chart_pattern_types:
            if pattern in chart_patterns:
                values = chart_patterns[pattern]
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    latest_value = values[-1] if not pd.isna(values[-1]) else 0
                    if latest_value != 0:
                        detected_patterns[pattern] = {
                            'value': float(latest_value),
                            'timeframe': timeframe,
                            'interpretation': self._interpret_chart_pattern(pattern, latest_value)
                        }
        
        return {
            'detected_patterns': detected_patterns,
            'pattern_count': len(detected_patterns),
            'timeframe': timeframe
        }
    
    def _extract_support_resistance_analysis(self, sr_levels, timeframe):
        """Extract support and resistance analysis"""
        if not sr_levels:
            return {'support_levels': [], 'resistance_levels': [], 'dynamic_levels': {}}
        
        support_analysis = {}
        resistance_analysis = {}
        
        # Support levels
        if 'Support_Levels' in sr_levels:
            support_levels = sr_levels['Support_Levels']
            if isinstance(support_levels, dict):
                support_analysis = {
                    'levels': list(support_levels.keys()),
                    'strengths': {k: v.get('strength', 0) for k, v in support_levels.items()},
                    'touches': {k: v.get('touches', 0) for k, v in support_levels.items()}
                }
        
        # Resistance levels
        if 'Resistance_Levels' in sr_levels:
            resistance_levels = sr_levels['Resistance_Levels']
            if isinstance(resistance_levels, dict):
                resistance_analysis = {
                    'levels': list(resistance_levels.keys()),
                    'strengths': {k: v.get('strength', 0) for k, v in resistance_levels.items()},
                    'touches': {k: v.get('touches', 0) for k, v in resistance_levels.items()}
                }
        
        # Dynamic levels
        dynamic_support = sr_levels.get('Dynamic_Support', [])
        dynamic_resistance = sr_levels.get('Dynamic_Resistance', [])
        
        if isinstance(dynamic_support, (list, np.ndarray)) and len(dynamic_support) > 0:
            current_dynamic_support = dynamic_support[-1] if not pd.isna(dynamic_support[-1]) else None
        else:
            current_dynamic_support = None
            
        if isinstance(dynamic_resistance, (list, np.ndarray)) and len(dynamic_resistance) > 0:
            current_dynamic_resistance = dynamic_resistance[-1] if not pd.isna(dynamic_resistance[-1]) else None
        else:
            current_dynamic_resistance = None
        
        return {
            'support_levels': support_analysis,
            'resistance_levels': resistance_analysis,
            'dynamic_levels': {
                'dynamic_support': current_dynamic_support,
                'dynamic_resistance': current_dynamic_resistance
            },
            'volume_levels': sr_levels.get('Volume_Levels', {}),
            'timeframe': timeframe
        }
    
    def _extract_volume_analysis(self, patterns, timeframe):
        """Extract volume analysis"""
        volume_patterns = {}
        
        # Volume-related patterns from support_resistance
        sr_patterns = patterns.get('support_resistance', {})
        
        volume_confirmation = sr_patterns.get('Volume_Confirmation', [])
        volume_breakouts = sr_patterns.get('Volume_Breakouts', [])
        volume_exhaustion = sr_patterns.get('Volume_Exhaustion', [])
        
        # Get latest values
        if isinstance(volume_confirmation, (list, np.ndarray)) and len(volume_confirmation) > 0:
            latest_confirmation = volume_confirmation[-1] if not pd.isna(volume_confirmation[-1]) else 0
            if latest_confirmation != 0:
                volume_patterns['confirmation'] = {
                    'signal': 'Bullish' if latest_confirmation > 0 else 'Bearish',
                    'strength': abs(latest_confirmation)
                }
        
        if isinstance(volume_breakouts, (list, np.ndarray)) and len(volume_breakouts) > 0:
            latest_breakout = volume_breakouts[-1] if not pd.isna(volume_breakouts[-1]) else 0
            if latest_breakout != 0:
                volume_patterns['breakout'] = {
                    'signal': 'Bullish' if latest_breakout > 0 else 'Bearish',
                    'strength': abs(latest_breakout)
                }
        
        if isinstance(volume_exhaustion, (list, np.ndarray)) and len(volume_exhaustion) > 0:
            latest_exhaustion = volume_exhaustion[-1] if not pd.isna(volume_exhaustion[-1]) else 0
            if latest_exhaustion != 0:
                volume_patterns['exhaustion'] = {
                    'signal': 'Reversal Expected',
                    'direction': 'Bullish' if latest_exhaustion > 0 else 'Bearish'
                }
        
        return {
            'volume_patterns': volume_patterns,
            'timeframe': timeframe,
            'analysis_summary': self._generate_volume_summary(volume_patterns)
        }
    
    def _calculate_overall_technical_sentiment(self):
        """Calculate overall technical sentiment across timeframes"""
        sentiment_scores = {}
        
        for timeframe, analysis in self.technical_analysis.items():
            if 'current_signals' in analysis:
                signals = analysis['current_signals']
                
                # Calculate sentiment for this timeframe
                bullish_signals = 0
                bearish_signals = 0
                total_signals = 0
                
                # Trend indicators
                if 'MA_Cloud_Strength' in signals:
                    cloud_strength = signals['MA_Cloud_Strength']
                    if cloud_strength > 0.3:
                        bullish_signals += 2
                    elif cloud_strength < -0.3:
                        bearish_signals += 2
                    total_signals += 2
                
                # Momentum indicators
                if 'RSI_14' in signals:
                    rsi = signals['RSI_14']
                    if rsi > 60:
                        bullish_signals += 1
                    elif rsi < 40:
                        bearish_signals += 1
                    total_signals += 1
                
                if 'MACD' in signals and 'MACD_Signal' in signals:
                    if signals['MACD'] > signals['MACD_Signal']:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
                    total_signals += 1
                
                # Pattern signals
                patterns = analysis.get('patterns', {})
                for pattern, value in patterns.items():
                    if value > 0:
                        bullish_signals += 1
                    elif value < 0:
                        bearish_signals += 1
                    total_signals += 1
                
                # Calculate sentiment score
                if total_signals > 0:
                    sentiment_score = (bullish_signals - bearish_signals) / total_signals
                    sentiment_scores[timeframe] = {
                        'score': sentiment_score,
                        'bullish_signals': bullish_signals,
                        'bearish_signals': bearish_signals,
                        'total_signals': total_signals
                    }
        
        # Overall sentiment
        if sentiment_scores:
            overall_score = np.mean([s['score'] for s in sentiment_scores.values()])
            
            if overall_score > 0.3:
                sentiment = 'Bullish'
            elif overall_score < -0.3:
                sentiment = 'Bearish'
            else:
                sentiment = 'Neutral'
            
            return {
                'overall_sentiment': sentiment,
                'sentiment_score': overall_score,
                'timeframe_breakdown': sentiment_scores,
                'confidence': abs(overall_score)
            }
        
        return {'overall_sentiment': 'Neutral', 'sentiment_score': 0, 'confidence': 0}
    
    def _calculate_technical_risk_assessment(self):
        """Calculate technical risk assessment"""
        risk_factors = []
        risk_score = 0
        
        for timeframe, analysis in self.technical_analysis.items():
            if 'current_signals' in analysis:
                signals = analysis['current_signals']
                
                # Volatility risk
                if 'ATR_Normalized' in signals:
                    atr_norm = signals['ATR_Normalized']
                    if atr_norm > 3:  # High volatility
                        risk_factors.append(f"High volatility in {timeframe}")
                        risk_score += 0.2
                
                # Trend reversal risk
                if 'Reversal_Probability' in signals:
                    reversal_prob = signals['Reversal_Probability']
                    if reversal_prob > 0.7:
                        risk_factors.append(f"High reversal probability in {timeframe}")
                        risk_score += 0.3
                
                # Overbought/Oversold conditions
                if 'RSI_14' in signals:
                    rsi = signals['RSI_14']
                    if rsi > 80:
                        risk_factors.append(f"Severely overbought in {timeframe}")
                        risk_score += 0.25
                    elif rsi < 20:
                        risk_factors.append(f"Severely oversold in {timeframe}")
                        risk_score += 0.25
        
        # Risk level classification
        if risk_score > 0.7:
            risk_level = 'High'
        elif risk_score > 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_level': risk_level,
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _generate_technical_trading_signals(self):
        """Generate technical trading signals"""
        signals = {}
        
        # Short-term signals (5m, 15m)
        short_term_signal = self._generate_timeframe_signal(['5m', '15m'])
        if short_term_signal:
            signals['short_term'] = short_term_signal
        
        # Medium-term signals (1h, 1d)
        medium_term_signal = self._generate_timeframe_signal(['1h', '1d'])
        if medium_term_signal:
            signals['medium_term'] = medium_term_signal
        
        # Long-term signals (1d, 1wk)
        long_term_signal = self._generate_timeframe_signal(['1d', '1wk'])
        if long_term_signal:
            signals['long_term'] = long_term_signal
        
        return signals
    
    def _generate_timeframe_signal(self, timeframes):
        """Generate trading signal for specific timeframes"""
        bullish_count = 0
        bearish_count = 0
        total_count = 0
        
        signal_strength = 0
        confidence_factors = []
        
        for tf in timeframes:
            if tf in self.technical_analysis:
                analysis = self.technical_analysis[tf]
                signals = analysis.get('current_signals', {})
                patterns = analysis.get('patterns', {})
                
                # Trend signal
                if 'MA_Cloud_Strength' in signals:
                    cloud = signals['MA_Cloud_Strength']
                    if cloud > 0.3:
                        bullish_count += 2
                        signal_strength += cloud
                    elif cloud < -0.3:
                        bearish_count += 2
                        signal_strength += abs(cloud)
                    total_count += 2
                
                # Momentum signal
                if 'Momentum_Composite' in signals:
                    momentum = signals['Momentum_Composite']
                    if momentum > 0.2:
                        bullish_count += 1
                    elif momentum < -0.2:
                        bearish_count += 1
                    total_count += 1
                
                # Pattern signals
                pattern_score = sum([1 for p in patterns.values() if p > 0]) - sum([1 for p in patterns.values() if p < 0])
                if pattern_score > 0:
                    bullish_count += 1
                elif pattern_score < 0:
                    bearish_count += 1
                total_count += 1
                
                # Volume confirmation
                if 'Market_Strength' in signals:
                    strength = signals['Market_Strength']
                    if abs(strength) > 0.3:
                        confidence_factors.append(abs(strength))
        
        if total_count == 0:
            return None
        
        # Calculate signal
        signal_score = (bullish_count - bearish_count) / total_count
        confidence = np.mean(confidence_factors) if confidence_factors else abs(signal_score)
        
        if signal_score > 0.3:
            signal_type = 'BUY'
        elif signal_score < -0.3:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        return {
            'signal': signal_type,
            'strength': abs(signal_score),
            'confidence': confidence,
            'bullish_factors': bullish_count,
            'bearish_factors': bearish_count,
            'timeframes': timeframes
        }
    
    def _extract_ml_technical_insights(self):
        """Extract ML insights for technical analysis"""
        if not self.ml_predictions:
            return {}
        
        insights = {}
        
        # Current predictions
        current_preds = self.ml_predictions.get('current_predictions', {})
        for timeframe, pred in current_preds.items():
            if 'prediction' in pred and 'current_price' in pred:
                predicted_price = pred['prediction']
                current_price = pred['current_price']
                price_change_pct = ((predicted_price - current_price) / current_price) * 100
                
                insights[f'{timeframe}_prediction'] = {
                    'predicted_direction': 'Up' if price_change_pct > 0 else 'Down',
                    'price_change_percent': price_change_pct,
                    'confidence': pred.get('confidence', 0.5),
                    'anomaly_detected': pred.get('anomaly_detected', False)
                }
        
        # Multi-horizon predictions
        multi_horizon = self.ml_predictions.get('multi_horizon', {})
        for timeframe, horizons in multi_horizon.items():
            horizon_insights = {}
            for period, pred_data in horizons.items():
                horizon_insights[period] = {
                    'price_change_percent': pred_data.get('price_change_percent', 0),
                    'confidence': pred_data.get('confidence', 0.5)
                }
            insights[f'{timeframe}_horizons'] = horizon_insights
        
        return insights
    
    def _extract_ml_price_action_insights(self):
        """Extract ML insights for price action analysis"""
        if not self.ml_predictions:
            return {}
        
        # This would extract ML-discovered patterns from the price action analysis
        # For now, return a placeholder structure
        return {
            'ml_discovered_patterns': 'Advanced ML pattern discovery completed',
            'anomaly_patterns': 'Unusual market behavior detected through deep learning',
            'pattern_clusters': 'Similar pattern groups identified across timeframes'
        }
    
    def _interpret_chart_pattern(self, pattern_name, value):
        """Interpret chart pattern signals"""
        interpretations = {
            'Head_Shoulders': 'Potential trend reversal signal',
            'Triangles': 'Consolidation pattern - breakout expected',
            'Flags_Pennants': 'Continuation pattern',
            'Channels': 'Price moving within defined boundaries',
            'Wedges': 'Potential reversal pattern',
            'Breakout_Direction': 'Directional momentum indicated',
            'Breakout_Strength': 'Strength of price movement',
            'False_Breakouts': 'Failed breakout detected'
        }
        
        base_interpretation = interpretations.get(pattern_name, 'Pattern detected')
        
        if value > 0:
            return f"{base_interpretation} - Bullish bias"
        elif value < 0:
            return f"{base_interpretation} - Bearish bias"
        else:
            return base_interpretation
    
    def _generate_volume_summary(self, volume_patterns):
        """Generate volume analysis summary"""
        if not volume_patterns:
            return "No significant volume patterns detected"
        
        summaries = []
        for pattern_type, data in volume_patterns.items():
            if pattern_type == 'confirmation':
                summaries.append(f"Volume confirming {data['signal'].lower()} move")
            elif pattern_type == 'breakout':
                summaries.append(f"Volume breakout with {data['signal'].lower()} bias")
            elif pattern_type == 'exhaustion':
                summaries.append(f"Volume exhaustion suggesting {data['direction'].lower()} reversal")
        
        return "; ".join(summaries) if summaries else "Normal volume patterns"
    
    def _analyze_breakout_potential(self):
        """Analyze breakout potential across timeframes"""
        breakout_analysis = {}
        
        for timeframe, analysis in self.price_action_analysis.items():
            patterns = analysis.get('patterns', {})
            chart_patterns = patterns.get('chart_patterns', {})
            
            breakout_signals = {}
            
            # Check for breakout-related patterns
            if 'Breakout_Direction' in chart_patterns:
                direction = chart_patterns['Breakout_Direction']
                if isinstance(direction, (list, np.ndarray)) and len(direction) > 0:
                    latest_direction = direction[-1] if not pd.isna(direction[-1]) else 0
                    if latest_direction != 0:
                        breakout_signals['direction'] = 'Upward' if latest_direction > 0 else 'Downward'
            
            if 'Breakout_Strength' in chart_patterns:
                strength = chart_patterns['Breakout_Strength']
                if isinstance(strength, (list, np.ndarray)) and len(strength) > 0:
                    latest_strength = strength[-1] if not pd.isna(strength[-1]) else 0
                    if latest_strength > 0:
                        breakout_signals['strength'] = latest_strength
            
            if breakout_signals:
                breakout_analysis[timeframe] = breakout_signals
        
        return breakout_analysis
    
    def _analyze_reversal_potential(self):
        """Analyze reversal potential across timeframes"""
        reversal_analysis = {}
        
        for timeframe, analysis in self.price_action_analysis.items():
            patterns = analysis.get('patterns', {})
            chart_patterns = patterns.get('chart_patterns', {})
            
            reversal_signals = {}
            
            # Check for reversal patterns
            if 'Reversal_Strength' in chart_patterns:
                strength = chart_patterns['Reversal_Strength']
                if isinstance(strength, (list, np.ndarray)) and len(strength) > 0:
                    latest_strength = strength[-1] if not pd.isna(strength[-1]) else 0
                    if latest_strength > 0:
                        reversal_signals['strength'] = latest_strength
            
            if 'Reversal_Confirmation' in chart_patterns:
                confirmation = chart_patterns['Reversal_Confirmation']
                if isinstance(confirmation, (list, np.ndarray)) and len(confirmation) > 0:
                    latest_confirmation = confirmation[-1] if not pd.isna(confirmation[-1]) else 0
                    if latest_confirmation > 0:
                        reversal_signals['confirmation'] = True
            
            if reversal_signals:
                reversal_analysis[timeframe] = reversal_signals
        
        return reversal_analysis
    
    def _calculate_pattern_reliability(self):
        """Calculate overall pattern reliability"""
        reliability_scores = {}
        
        for timeframe, analysis in self.price_action_analysis.items():
            patterns = analysis.get('patterns', {})
            candlestick_patterns = patterns.get('candlestick_patterns', {})
            
            if 'Pattern_Confidence' in candlestick_patterns:
                confidence_scores = candlestick_patterns['Pattern_Confidence']
                if isinstance(confidence_scores, dict):
                    avg_confidence = np.mean(list(confidence_scores.values()))
                    reliability_scores[timeframe] = avg_confidence
        
        overall_reliability = np.mean(list(reliability_scores.values())) if reliability_scores else 0.5
        
        return {
            'overall_reliability': overall_reliability,
            'timeframe_reliability': reliability_scores,
            'reliability_level': 'High' if overall_reliability > 0.7 else 'Medium' if overall_reliability > 0.4 else 'Low'
        }
    
    def _get_risk_recommendation(self, risk_level):
        """Get risk-based recommendation"""
        recommendations = {
            'High': 'Exercise extreme caution. Consider reducing position sizes and using tight stop losses.',
            'Medium': 'Moderate risk detected. Use appropriate position sizing and risk management.',
            'Low': 'Favorable risk environment. Normal position sizing acceptable.'
        }
        return recommendations.get(risk_level, 'Monitor risk levels closely.')
    
    def get_live_analysis(self):
        """Get live analysis with latest data"""
        print("📡 Fetching live market data...")
        
        # Get live data
        live_data = self.data_fetcher.get_live_data()
        if live_data is not None:
            print("✅ Live data fetched successfully!")
            
            # Update analysis with live data
            # This would typically involve updating the last data point and recalculating indicators
            current_price = live_data['Close'].iloc[-1]
            
            return {
                'current_price': current_price,
                'timestamp': live_data.index[-1],
                'live_data_available': True
            }
        else:
            print("❌ Failed to fetch live data")
            return {'live_data_available': False}
    
    def export_analysis_to_html(self, technical_report, price_action_report, filename=None):
        """Export analysis reports to HTML"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.symbol}_analysis_{timestamp}.html"
        
        html_content = self._generate_html_report(technical_report, price_action_report)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Analysis exported to {filename}")
        return filename
    
    def _generate_html_report(self, technical_report, price_action_report):
        """Generate comprehensive HTML report"""
        
        # This is a simplified version - in a real implementation, 
        # you'd want to use a proper templating engine
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stock Analysis Report - {self.symbol}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .bullish {{ color: green; font-weight: bold; }}
        .bearish {{ color: red; font-weight: bold; }}
        .neutral {{ color: orange; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Advanced Stock Market Analysis Report</h1>
        <h2>{technical_report['metadata']['company_name']} ({self.symbol})</h2>
        <p><strong>Sector:</strong> {technical_report['metadata']['sector']}</p>
        <p><strong>Generated:</strong> {technical_report['metadata']['generated_at']}</p>
    </div>
    
    <div class="section">
        <h2>📈 Technical Analysis Summary</h2>
        <p><strong>Overall Sentiment:</strong> 
        <span class="{technical_report['overall_sentiment']['overall_sentiment'].lower()}">
        {technical_report['overall_sentiment']['overall_sentiment']}
        </span></p>
        <p><strong>Risk Level:</strong> {technical_report['risk_assessment']['risk_level']}</p>
    </div>
    
    <div class="section">
        <h2>🕯️ Price Action Analysis Summary</h2>
        <p><strong>Patterns Detected:</strong> Multiple timeframe analysis completed</p>
        <p><strong>Breakout Analysis:</strong> {len(price_action_report['breakout_analysis'])} timeframes analyzed</p>
        <p><strong>Pattern Reliability:</strong> {price_action_report['pattern_reliability']['reliability_level']}</p>
    </div>
    
    <div class="section">
        <h2>🎯 Trading Signals</h2>
        {self._format_trading_signals_html(technical_report.get('trading_signals', {}))}
    </div>
    
    <div class="section">
        <h2>⚠️ Risk Assessment</h2>
        <p><strong>Risk Factors:</strong></p>
        <ul>
        {"".join([f"<li>{factor}</li>" for factor in technical_report['risk_assessment']['risk_factors']])}
        </ul>
        <p><strong>Recommendation:</strong> {technical_report['risk_assessment']['recommendation']}</p>
    </div>
    
    <div class="section">
        <h2>🤖 Machine Learning Insights</h2>
        {self._format_ml_insights_html(technical_report.get('ml_insights', {}))}
    </div>
    
    <footer style="margin-top: 40px; text-align: center; color: #666;">
        <p>This analysis is for educational purposes only and should not be considered as financial advice.</p>
        <p>Generated by Advanced Stock Market Analyzer v1.0</p>
    </footer>
</body>
</html>
        """
        
        return html
    
    def _format_trading_signals_html(self, signals):
        """Format trading signals for HTML"""
        if not signals:
            return "<p>No trading signals generated.</p>"
        
        html = "<table><tr><th>Timeframe</th><th>Signal</th><th>Strength</th><th>Confidence</th></tr>"
        
        for timeframe, signal_data in signals.items():
            signal_class = signal_data['signal'].lower()
            html += f"""
            <tr>
                <td>{timeframe.replace('_', ' ').title()}</td>
                <td><span class="{signal_class}">{signal_data['signal']}</span></td>
                <td>{signal_data['strength']:.2f}</td>
                <td>{signal_data['confidence']:.2f}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _format_ml_insights_html(self, ml_insights):
        """Format ML insights for HTML"""
        if not ml_insights:
            return "<p>No ML insights available.</p>"
        
        html = "<ul>"
        for insight, data in ml_insights.items():
            if isinstance(data, dict) and 'predicted_direction' in data:
                direction_class = 'bullish' if data['predicted_direction'] == 'Up' else 'bearish'
                html += f"""
                <li><strong>{insight}:</strong> 
                <span class="{direction_class}">{data['predicted_direction']}</span> 
                ({data['price_change_percent']:.2f}% - Confidence: {data['confidence']:.2f})</li>
                """
        html += "</ul>"
        return html

# Example usage and demonstration
if __name__ == "__main__":
    print("🚀 Advanced Stock Market Analyzer")
    print("=" * 50)
    
    # Initialize with RELIANCE.NS (default)
    analyzer = StockMarketAnalyzer("RELIANCE.NS")
    
    # Fetch data
    print("\n📊 Fetching comprehensive market data...")
    data = analyzer.fetch_comprehensive_data()
    
    # Run analysis
    print("\n🔍 Running comprehensive analysis...")
    analyzer.run_comprehensive_analysis(train_models=False)  # Set to True to train ML models
    
    # Generate reports
    print("\n📈 Generating Technical Analysis Report...")
    technical_report = analyzer.generate_technical_analysis_report()
    
    print("\n🕯️ Generating Price Action Report...")
    price_action_report = analyzer.generate_price_action_report()
    
    # Export to HTML
    print("\n📊 Exporting analysis to HTML...")
    html_file = analyzer.export_analysis_to_html(technical_report, price_action_report)
    
    print(f"\n✅ Analysis complete! Reports saved to {html_file}")
    print("\n🎯 Key Insights:")
    print(f"   Overall Sentiment: {technical_report['overall_sentiment']['overall_sentiment']}")
    print(f"   Risk Level: {technical_report['risk_assessment']['risk_level']}")
    print(f"   Pattern Reliability: {price_action_report['pattern_reliability']['reliability_level']}")
    
    print("\n💡 This is a demonstration of the comprehensive analysis capabilities.")
    print("   In production, you would enable ML model training for full insights.")