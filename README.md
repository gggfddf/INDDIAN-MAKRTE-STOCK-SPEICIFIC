# 🚀 Advanced Deep Learning Stock Market Analysis Module

A comprehensive, expert-level deep learning system for Indian stock market analysis that unveils market secrets and provides detailed insights through advanced technical analysis, price action analysis, and machine learning predictions.

## 🌟 Features

### 📊 **Multi-Timeframe Data Analysis**
- **5-minute, 15-minute, 1-hour, 1-day, and 1-week** analysis
- **Live data fetching** from Yahoo Finance and NSE APIs
- **Maximum historical range** for robust analysis
- **Automatic data validation** and cleaning

### 🔧 **Advanced Technical Analysis (40+ Indicators)**
- **Moving Averages**: SMA, EMA, MA Cloud Analysis, Golden/Death Cross
- **Bollinger Bands**: Squeeze detection, expansion patterns, band walking
- **VWAP**: Flatness detection, reversion patterns, multi-timeframe analysis
- **RSI**: Divergence patterns, trend analysis, overbought/oversold clusters
- **MACD**: Histogram patterns, signal line analysis, trend strength
- **Stochastic**: K/D crossover patterns, divergence analysis
- **Volume Indicators**: OBV patterns, volume price trend, accumulation/distribution
- **Momentum Indicators**: ROC patterns, momentum divergence, Williams %R
- **Volatility Indicators**: ATR patterns, volatility breakout signals
- **Custom Composite Indicators**: Multi-indicator trend scores, market strength

### 🕯️ **Comprehensive Price Action Analysis**
- **Traditional Candlestick Patterns**: All major patterns (Doji, Hammer, Engulfing, etc.)
- **ML-Discovered Patterns**: Machine learning identifies new candlestick patterns
- **Chart Pattern Recognition**: Head & Shoulders, Triangles, Flags, Pennants, Channels
- **Support/Resistance Analysis**: Dynamic levels with strength indicators
- **Gap Analysis**: Gap detection, classification, and fill probability
- **Volume-Price Relationships**: Volume confirmation, breakout patterns
- **Pattern Confidence Scoring**: Historical success rates for each pattern

### 🤖 **Deep Learning Models**
- **LSTM Networks**: Sequential pattern recognition in price and volume data
- **CNN Layers**: Chart pattern recognition and visual pattern detection
- **Attention Mechanisms**: Focus on most relevant features for prediction
- **Hybrid Models**: Combined CNN-LSTM architectures
- **Autoencoders**: Anomaly detection in price patterns
- **Ensemble Methods**: Multiple models for robust predictions
- **Traditional ML**: XGBoost and LightGBM for feature importance

### 📈 **Intelligent Reporting**
- **Separate Technical and Price Action Reports**
- **HTML Export** with interactive visualizations
- **JSON Export** for programmatic access
- **Confidence Scores** for all predictions
- **Risk Assessment** with actionable recommendations
- **Trading Signals** across multiple timeframes

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd stock-market-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from stock_market_analyzer import StockMarketAnalyzer

# Initialize analyzer (default: RELIANCE.NS)
analyzer = StockMarketAnalyzer("RELIANCE.NS")

# Fetch comprehensive data
data = analyzer.fetch_comprehensive_data()

# Run complete analysis
analyzer.run_comprehensive_analysis(train_models=False)

# Generate reports
technical_report = analyzer.generate_technical_analysis_report()
price_action_report = analyzer.generate_price_action_report()

# Export to HTML
analyzer.export_analysis_to_html(technical_report, price_action_report)
```

### Command Line Usage

```bash
# Basic analysis
python example_usage.py RELIANCE.NS

# With ML training (resource intensive)
python example_usage.py TCS.NS --ml

# List available stocks
python example_usage.py --list

# Help
python example_usage.py --help
```

## 📋 Supported Stocks

The system supports all NSE-listed stocks. Popular stocks include:

- **RELIANCE.NS** - Reliance Industries Ltd.
- **TCS.NS** - Tata Consultancy Services Ltd.
- **INFY.NS** - Infosys Ltd.
- **HDFCBANK.NS** - HDFC Bank Ltd.
- **ICICIBANK.NS** - ICICI Bank Ltd.
- **HINDUNILVR.NS** - Hindustan Unilever Ltd.
- **ITC.NS** - ITC Ltd.
- **SBIN.NS** - State Bank of India
- **BHARTIARTL.NS** - Bharti Airtel Ltd.
- **KOTAKBANK.NS** - Kotak Mahindra Bank Ltd.

*Use `.NS` suffix for NSE stocks*

## 🏗️ System Architecture

```
📦 Stock Market Analyzer
├── 📊 data_fetcher.py          # Multi-source data fetching
├── 🔧 technical_indicators.py  # 40+ technical indicators
├── 🕯️ price_action_analysis.py # Candlestick & chart patterns
├── 🤖 deep_learning_models.py  # ML models & predictions
├── 🎯 stock_market_analyzer.py # Main analysis orchestrator
├── 📋 example_usage.py         # Usage examples
└── 📄 requirements.txt         # Dependencies
```

## 🔍 Technical Analysis Features

### Trend Analysis
- **MA Cloud Strength**: Multi-timeframe moving average alignment
- **Golden/Death Cross**: Major trend reversal signals
- **Dynamic Support/Resistance**: Moving average-based levels
- **Trend Strength**: Quantified trend momentum

### Momentum Analysis
- **RSI Divergence**: Price vs momentum divergence detection
- **MACD Patterns**: Histogram and signal line analysis
- **ROC Analysis**: Rate of change patterns and crossovers
- **Composite Momentum**: Multi-indicator momentum score

### Volatility Analysis
- **ATR Patterns**: Average True Range trend analysis
- **Bollinger Band Dynamics**: Squeeze, expansion, walking patterns
- **Volatility Regimes**: High/low volatility classification
- **Breakout Signals**: Volatility-based breakout detection

## 🕯️ Price Action Features

### Candlestick Patterns
- **Single Patterns**: Doji, Hammer, Shooting Star, Marubozu
- **Double Patterns**: Engulfing, Harami, Piercing, Dark Cloud
- **Triple Patterns**: Morning/Evening Star, Three Soldiers/Crows
- **Advanced Patterns**: Abandoned Baby, Belt Hold, Breakaway

### Chart Patterns
- **Reversal Patterns**: Head & Shoulders, Double Tops/Bottoms
- **Continuation Patterns**: Triangles, Flags, Pennants, Channels
- **Breakout Patterns**: Wedges, Rectangle breakouts
- **Pattern Completion**: Probability of pattern completion

### ML Pattern Discovery
- **Clustering Analysis**: Similar pattern identification
- **Anomaly Detection**: Unusual pattern recognition
- **Sequence Mining**: Frequent candlestick sequences
- **Pattern Evolution**: How patterns change across timeframes

## 🤖 Machine Learning Features

### Model Architecture
- **LSTM Networks**: 3-layer LSTM with dropout and batch normalization
- **CNN Models**: Multi-layer convolutions for pattern recognition
- **Attention Models**: Multi-head attention for feature focus
- **Hybrid Models**: Combined CNN-LSTM architectures
- **Autoencoders**: Unsupervised anomaly detection

### Training Features
- **Time Series Validation**: Walk-forward analysis
- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rates
- **Ensemble Voting**: Multiple model consensus
- **Feature Engineering**: 100+ engineered features

### Prediction Capabilities
- **Multi-Horizon Forecasting**: 1-5 period predictions
- **Confidence Intervals**: Statistical confidence measures
- **Anomaly Detection**: Unusual market behavior identification
- **Feature Attribution**: Which features drive predictions

## 📊 Output Reports

### Technical Analysis Report
```json
{
  "metadata": {
    "symbol": "RELIANCE.NS",
    "company_name": "Reliance Industries Ltd.",
    "generated_at": "2024-01-15T10:30:00"
  },
  "overall_sentiment": {
    "sentiment": "Bullish",
    "confidence": 0.78
  },
  "trading_signals": {
    "short_term": {"signal": "BUY", "strength": 0.65},
    "medium_term": {"signal": "HOLD", "strength": 0.45}
  },
  "risk_assessment": {
    "risk_level": "Medium",
    "risk_factors": ["High volatility in 5m timeframe"]
  }
}
```

### Price Action Report
```json
{
  "candlestick_analysis": {
    "1d": {
      "detected_patterns": {
        "Hammer": {"signal": "Bullish", "strength": 100}
      },
      "total_patterns_detected": 3,
      "bullish_patterns": 2,
      "bearish_patterns": 1
    }
  },
  "support_resistance": {
    "1d": {
      "support_levels": [2450.5, 2420.0],
      "resistance_levels": [2480.0, 2495.5],
      "dynamic_support": 2465.2
    }
  }
}
```

## ⚙️ Configuration

### Data Sources
- **Primary**: Yahoo Finance (yfinance)
- **Secondary**: NSE API (nsepy)
- **Live Data**: Real-time price updates
- **Historical Range**: Maximum available (up to 20+ years)

### Timeframes
- **5m**: 60 days of 5-minute data
- **15m**: 60 days of 15-minute data  
- **1h**: 2 years of hourly data
- **1d**: Maximum daily data available
- **1wk**: Maximum weekly data available

### Model Parameters
- **Sequence Length**: 60 periods for LSTM models
- **Training Split**: 80% training, 20% validation
- **Batch Size**: 32 for deep learning models
- **Epochs**: Up to 100 with early stopping

## 🔬 Advanced Features

### Pattern Confidence Scoring
Each detected pattern includes:
- **Historical Success Rate**: Based on past performance
- **Volume Confirmation**: Volume support for the pattern
- **Context Confirmation**: Alignment with trend and market conditions
- **Combined Confidence**: Weighted average of all factors

### Multi-Timeframe Analysis
- **Cross-Timeframe Validation**: Patterns confirmed across timeframes
- **Timeframe Hierarchy**: Higher timeframes override lower ones
- **Confluence Detection**: Multiple signals from different timeframes
- **Timeframe-Specific Signals**: Tailored for different trading styles

### Risk Management
- **Volatility Assessment**: Current vs historical volatility
- **Trend Reversal Probability**: Likelihood of trend changes
- **Extreme Conditions**: Overbought/oversold warnings
- **Dynamic Risk Scoring**: Real-time risk level updates

## 📈 Performance Metrics

### Model Accuracy
- **Directional Accuracy**: Prediction of price direction
- **RMSE/MAE**: Price prediction error metrics
- **Sharpe Ratio**: Risk-adjusted return simulation
- **Maximum Drawdown**: Worst-case scenario analysis

### Pattern Success Rates
- **Bullish Patterns**: Historical success rates for upward moves
- **Bearish Patterns**: Historical success rates for downward moves
- **Neutral Patterns**: Consolidation and continuation rates
- **False Signal Detection**: Identification of failed patterns

## 🛠️ Development

### Adding New Indicators
```python
def _calculate_custom_indicator(self):
    """Add custom technical indicator"""
    # Your indicator calculation
    custom_values = your_calculation(self.close)
    self.indicators['Custom_Indicator'] = custom_values
```

### Adding New Patterns
```python
def _detect_custom_pattern(self):
    """Add custom candlestick pattern"""
    pattern = np.zeros(len(self.close))
    # Your pattern detection logic
    return pattern
```

### Extending ML Models
```python
def build_custom_model(self, input_shape):
    """Add custom deep learning model"""
    # Your model architecture
    return model
```

## 📊 Example Output

When you run the analyzer, you get:

### Console Output
```
🚀 Advanced Stock Market Analysis System
============================================================
📊 Analyzing: RELIANCE.NS
🤖 ML Training: Disabled (for demo)
============================================================

📡 Step 1: Fetching Market Data
----------------------------------------
✅ Data fetched successfully!
📈 Company: Reliance Industries Ltd.
🏭 Sector: Energy
💰 Market Cap: ₹18,50,000,00,00,000

📈 TECHNICAL ANALYSIS INSIGHTS
   ───────────────────────────────────
   Overall Sentiment: 🟢 Bullish (0.65)
   Risk Level: 🟡 Medium (0.45)
   Trading Signals:
     Short Term: 🟢 BUY (Strength: 0.72, Confidence: 0.68)
     Medium Term: 🟡 HOLD (Strength: 0.45, Confidence: 0.52)

🕯️ PRICE ACTION ANALYSIS INSIGHTS
   ──────────────────────────────────────
   Pattern Reliability: 🟢 High (0.78)
   Candlestick Patterns: 5 detected
     🟢 Bullish: 3
     🔴 Bearish: 2
   Breakout Signals: 2 timeframes
     1d: 🟢 Upward (Strength: 1.25)
```

### Generated Files
- **HTML Report**: `RELIANCE.NS_analysis_20240115_103000.html`
- **Technical JSON**: `RELIANCE.NS_technical_analysis_20240115_103000.json`
- **Price Action JSON**: `RELIANCE.NS_price_action_20240115_103000.json`

## 🔒 Disclaimer

⚠️ **Important**: This analysis system is for **educational and research purposes only**. 

- **Not Financial Advice**: All outputs should be considered educational content
- **Market Risk**: Trading involves substantial risk of loss
- **Professional Consultation**: Always consult qualified financial advisors
- **Past Performance**: Historical analysis doesn't guarantee future results
- **Use at Own Risk**: Users are responsible for their trading decisions

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **New Technical Indicators**: Add more sophisticated indicators
- **Advanced ML Models**: Implement transformer architectures
- **Real-time Analysis**: Add live streaming analysis
- **Mobile Interface**: Create mobile-friendly interfaces
- **Backtesting Engine**: Add comprehensive backtesting capabilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing comprehensive market data
- **NSE**: For Indian stock market data
- **TA-Lib**: For technical analysis functions
- **TensorFlow**: For deep learning capabilities
- **Scikit-learn**: For machine learning tools

## 📞 Support

For questions, issues, or contributions:

1. **Create an Issue**: For bugs or feature requests
2. **Discussion**: For general questions about usage
3. **Pull Requests**: For code contributions

---

**Built with ❤️ for the Indian stock market trading community**

*"Unveiling market secrets through advanced analytics and machine learning"*
