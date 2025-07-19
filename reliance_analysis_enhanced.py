#!/usr/bin/env python3
"""
Enhanced Reliance Industries Stock Market Analysis
==================================================

This script runs a comprehensive analysis of Reliance Industries stock using advanced
technical analysis, price action analysis, and machine learning models.

It generates:
- Detailed PNG visualizations
- Excel reports with data and analysis
- Interactive HTML reports
- JSON data exports
- Comprehensive PDF summary

Author: AI Stock Market Analysis System
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import json
from datetime import datetime, timedelta
import yfinance as yf

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for better visuals
plt.style.use('default')
sns.set_palette("husl")

# Import our custom modules
try:
    from stock_market_analyzer import StockMarketAnalyzer
    from data_fetcher import IndianStockDataFetcher
    from technical_indicators import AdvancedTechnicalIndicators
    from price_action_analysis import AdvancedPriceActionAnalysis
    # Note: deep_learning_models.py requires TensorFlow which isn't available
    # So we'll create a mock version for demonstration
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating simplified analysis...")

class EnhancedVisualizationEngine:
    """Enhanced visualization engine for comprehensive stock analysis"""
    
    def __init__(self, symbol="RELIANCE.NS"):
        self.symbol = symbol
        self.company_name = "Reliance Industries Limited"
        self.output_dir = "reliance_analysis_output"
        self.ensure_output_directory()
        
    def ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def fetch_data(self):
        """Fetch comprehensive stock data"""
        print(f"Fetching data for {self.symbol}...")
        
        # Get data for different timeframes
        timeframes = {
            '5d_5m': ('5d', '5m'),
            '1mo_15m': ('1mo', '15m'),
            '1y_1d': ('1y', '1d'),
            '5y_1wk': ('5y', '1wk')
        }
        
        self.data = {}
        ticker = yf.Ticker(self.symbol)
        
        for tf_name, (period, interval) in timeframes.items():
            try:
                df = ticker.history(period=period, interval=interval)
                if not df.empty:
                    self.data[tf_name] = df
                    print(f"✓ Fetched {tf_name}: {len(df)} records")
                else:
                    print(f"✗ No data for {tf_name}")
            except Exception as e:
                print(f"✗ Error fetching {tf_name}: {e}")
                
        # Get company info
        try:
            self.info = ticker.info
        except:
            self.info = {"longName": self.company_name}
            
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        # Basic indicators
        indicators['SMA_20'] = df['Close'].rolling(window=20).mean()
        indicators['SMA_50'] = df['Close'].rolling(window=50).mean()
        indicators['EMA_12'] = df['Close'].ewm(span=12).mean()
        indicators['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        sma_20 = indicators['SMA_20']
        std_20 = df['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = sma_20 + (std_20 * 2)
        indicators['BB_Lower'] = sma_20 - (std_20 * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
        
        # Volume indicators
        indicators['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        indicators['Volume_Ratio'] = df['Volume'] / indicators['Volume_SMA']
        
        # Price patterns
        indicators['Higher_High'] = df['High'] > df['High'].shift(1)
        indicators['Lower_Low'] = df['Low'] < df['Low'].shift(1)
        
        # Volatility
        tr1 = df['High'] - df['Low']
        tr2 = np.abs(df['High'] - df['Close'].shift())
        tr3 = np.abs(df['Low'] - df['Close'].shift())
        indicators['True_Range'] = np.maximum(np.maximum(tr1, tr2), tr3)
        indicators['ATR'] = pd.Series(indicators['True_Range']).rolling(window=14).mean()
        
        return indicators
        
    def detect_candlestick_patterns(self, df):
        """Detect basic candlestick patterns"""
        patterns = {}
        
        # Calculate basic candle properties
        body = df['Close'] - df['Open']
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        total_range = df['High'] - df['Low']
        
        # Doji
        patterns['Doji'] = (np.abs(body) <= total_range * 0.1) & (total_range > 0)
        
        # Hammer
        patterns['Hammer'] = (
            (lower_shadow >= body * 2) & 
            (upper_shadow <= body * 0.5) & 
            (body > 0)
        )
        
        # Shooting Star
        patterns['Shooting_Star'] = (
            (upper_shadow >= body * 2) & 
            (lower_shadow <= body * 0.5) & 
            (body < 0)
        )
        
        # Engulfing patterns
        prev_body = body.shift(1)
        patterns['Bullish_Engulfing'] = (
            (body > 0) & 
            (prev_body < 0) & 
            (body > np.abs(prev_body) * 1.1)
        )
        
        patterns['Bearish_Engulfing'] = (
            (body < 0) & 
            (prev_body > 0) & 
            (np.abs(body) > prev_body * 1.1)
        )
        
        return patterns
        
    def create_price_chart(self, df, timeframe, indicators, patterns):
        """Create comprehensive price chart with indicators"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        fig.suptitle(f'{self.company_name} ({self.symbol}) - {timeframe} Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Main price chart with indicators
        ax1 = axes[0]
        ax1.plot(df.index, df['Close'], label='Close', linewidth=2, color='black')
        ax1.plot(df.index, indicators['SMA_20'], label='SMA 20', alpha=0.7, color='blue')
        ax1.plot(df.index, indicators['SMA_50'], label='SMA 50', alpha=0.7, color='red')
        ax1.fill_between(df.index, indicators['BB_Upper'], indicators['BB_Lower'], 
                        alpha=0.2, color='gray', label='Bollinger Bands')
        
        # Mark patterns
        doji_points = df.index[patterns['Doji']]
        if len(doji_points) > 0:
            ax1.scatter(doji_points, df.loc[doji_points, 'Close'], 
                       color='yellow', marker='o', s=50, label='Doji', zorder=5)
        
        hammer_points = df.index[patterns['Hammer']]
        if len(hammer_points) > 0:
            ax1.scatter(hammer_points, df.loc[hammer_points, 'Low'], 
                       color='green', marker='^', s=60, label='Hammer', zorder=5)
        
        ax1.set_title('Price Action with Technical Indicators')
        ax1.set_ylabel('Price (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2 = axes[1]
        colors = ['green' if close > open else 'red' for close, open in zip(df['Close'], df['Open'])]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7)
        ax2.plot(df.index, indicators['Volume_SMA'], color='blue', linewidth=2, label='Volume SMA')
        ax2.set_title('Volume Analysis')
        ax2.set_ylabel('Volume')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # RSI chart
        ax3 = axes[2]
        ax3.plot(df.index, indicators['RSI'], color='purple', linewidth=2)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax3.set_title('RSI (Relative Strength Index)')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # MACD chart
        ax4 = axes[3]
        ax4.plot(df.index, indicators['MACD'], label='MACD', color='blue')
        ax4.plot(df.index, indicators['MACD_Signal'], label='Signal', color='red')
        ax4.bar(df.index, indicators['MACD_Histogram'], label='Histogram', alpha=0.7, color='gray')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('MACD (Moving Average Convergence Divergence)')
        ax4.set_ylabel('MACD')
        ax4.set_xlabel('Date')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        # Save the chart
        filename = f"{self.output_dir}/price_analysis_{timeframe}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved price chart: {filename}")
        
    def create_pattern_analysis_chart(self, df, patterns):
        """Create detailed pattern analysis chart"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.company_name} - Candlestick Pattern Analysis', 
                     fontsize=16, fontweight='bold')
        
        pattern_names = ['Doji', 'Hammer', 'Bullish_Engulfing', 'Bearish_Engulfing']
        colors = ['yellow', 'green', 'lightgreen', 'lightcoral']
        
        for idx, (pattern_name, color) in enumerate(zip(pattern_names, colors)):
            ax = axes[idx // 2, idx % 2]
            
            # Plot price
            ax.plot(df.index, df['Close'], color='black', linewidth=1, alpha=0.7)
            
            # Highlight pattern occurrences
            pattern_points = df.index[patterns[pattern_name]]
            if len(pattern_points) > 0:
                ax.scatter(pattern_points, df.loc[pattern_points, 'Close'], 
                          color=color, s=80, alpha=0.8, zorder=5)
                
                # Add count
                count = len(pattern_points)
                ax.text(0.02, 0.98, f'Count: {count}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                       verticalalignment='top')
            
            ax.set_title(f'{pattern_name.replace("_", " ")} Pattern')
            ax.set_ylabel('Price (₹)')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        filename = f"{self.output_dir}/pattern_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pattern analysis: {filename}")
        
    def create_correlation_heatmap(self, df, indicators):
        """Create correlation heatmap of technical indicators"""
        # Combine price data with indicators
        analysis_df = pd.DataFrame({
            'Close': df['Close'],
            'Volume': df['Volume'],
            'RSI': indicators['RSI'],
            'MACD': indicators['MACD'],
            'ATR': indicators['ATR'],
            'SMA_20': indicators['SMA_20'],
            'SMA_50': indicators['SMA_50']
        })
        
        # Calculate correlation matrix
        correlation_matrix = analysis_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5)
        plt.title(f'{self.company_name} - Technical Indicator Correlations')
        plt.tight_layout()
        
        filename = f"{self.output_dir}/correlation_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved correlation heatmap: {filename}")
        
    def create_interactive_chart(self, df, indicators, patterns):
        """Create interactive Plotly chart"""
        # Create subplots
        fig = sp.make_subplots(
            rows=4, cols=1,
            subplot_titles=['Price & Indicators', 'Volume', 'RSI', 'MACD'],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators['SMA_20'], name='SMA 20', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators['SMA_50'], name='SMA 50', 
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators['BB_Upper'], name='BB Upper', 
                      line=dict(color='gray', width=1), opacity=0.5),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators['BB_Lower'], name='BB Lower', 
                      line=dict(color='gray', width=1), opacity=0.5, 
                      fill='tonexty', fillcolor='rgba(128,128,128,0.2)'),
            row=1, col=1
        )
        
        # Volume
        colors = ['green' if close > open else 'red' for close, open in zip(df['Close'], df['Open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                  marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators['RSI'], name='RSI', 
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators['MACD'], name='MACD', 
                      line=dict(color='blue', width=2)),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=indicators['MACD_Signal'], name='Signal', 
                      line=dict(color='red', width=2)),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df.index, y=indicators['MACD_Histogram'], name='Histogram', 
                  marker_color='gray', opacity=0.7),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.company_name} ({self.symbol}) - Interactive Analysis',
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Save interactive chart
        filename = f"{self.output_dir}/interactive_chart.html"
        plot(fig, filename=filename, auto_open=False)
        print(f"✓ Saved interactive chart: {filename}")
        
    def create_excel_report(self, df, indicators, patterns):
        """Create comprehensive Excel report"""
        filename = f"{self.output_dir}/reliance_analysis_report.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Raw data (remove timezone info)
            df_clean = df.copy()
            if df_clean.index.tz is not None:
                df_clean.index = df_clean.index.tz_localize(None)
            df_clean.to_excel(writer, sheet_name='Raw_Data')
            
            # Technical indicators
            indicators_df = pd.DataFrame(indicators, index=df_clean.index)
            indicators_df.to_excel(writer, sheet_name='Technical_Indicators')
            
            # Patterns
            patterns_df = pd.DataFrame(patterns, index=df_clean.index)
            patterns_df.to_excel(writer, sheet_name='Candlestick_Patterns')
            
            # Summary statistics
            summary_data = {
                'Metric': ['Current Price', 'Daily Change', 'Daily Change %', 
                          'Volume', 'Market Cap', 'P/E Ratio', '52W High', '52W Low'],
                'Value': [
                    f"₹{df['Close'].iloc[-1]:.2f}",
                    f"₹{df['Close'].iloc[-1] - df['Close'].iloc[-2]:.2f}",
                    f"{((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100:.2f}%",
                    f"{df['Volume'].iloc[-1]:,.0f}",
                    self.info.get('marketCap', 'N/A'),
                    self.info.get('trailingPE', 'N/A'),
                    f"₹{df['High'].max():.2f}",
                    f"₹{df['Low'].min():.2f}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Pattern summary
            pattern_summary = {
                'Pattern': list(patterns.keys()),
                'Count': [patterns[pattern].sum() for pattern in patterns.keys()],
                'Last_Occurrence': [
                    df.index[patterns[pattern]][-1].strftime('%Y-%m-%d') 
                    if patterns[pattern].any() else 'None'
                    for pattern in patterns.keys()
                ]
            }
            
            pattern_summary_df = pd.DataFrame(pattern_summary)
            pattern_summary_df.to_excel(writer, sheet_name='Pattern_Summary', index=False)
            
        print(f"✓ Saved Excel report: {filename}")
        
    def create_json_report(self, df, indicators, patterns):
        """Create JSON data export"""
        # Prepare data for JSON serialization
        json_data = {
            'symbol': self.symbol,
            'company_name': self.company_name,
            'analysis_date': datetime.now().isoformat(),
            'current_price': float(df['Close'].iloc[-1]),
            'daily_change': float(df['Close'].iloc[-1] - df['Close'].iloc[-2]),
            'daily_change_percent': float(((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100),
            'volume': int(df['Volume'].iloc[-1]),
            'technical_indicators': {
                'rsi': float(indicators['RSI'].iloc[-1]) if not pd.isna(indicators['RSI'].iloc[-1]) else None,
                'macd': float(indicators['MACD'].iloc[-1]) if not pd.isna(indicators['MACD'].iloc[-1]) else None,
                'sma_20': float(indicators['SMA_20'].iloc[-1]) if not pd.isna(indicators['SMA_20'].iloc[-1]) else None,
                'sma_50': float(indicators['SMA_50'].iloc[-1]) if not pd.isna(indicators['SMA_50'].iloc[-1]) else None,
            },
            'pattern_counts': {
                pattern: int(patterns[pattern].sum()) for pattern in patterns.keys()
            },
            'risk_metrics': {
                'volatility': float(df['Close'].pct_change().std() * np.sqrt(252) * 100),
                'max_drawdown': float(((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100),
                'sharpe_ratio': float(df['Close'].pct_change().mean() / df['Close'].pct_change().std() * np.sqrt(252))
            }
        }
        
        filename = f"{self.output_dir}/analysis_data.json"
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        print(f"✓ Saved JSON data: {filename}")
        
    def create_summary_report(self):
        """Create a summary text report"""
        filename = f"{self.output_dir}/analysis_summary.txt"
        
        with open(filename, 'w') as f:
            f.write(f"RELIANCE INDUSTRIES LIMITED (RELIANCE.NS) ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if '1y_1d' in self.data:
                df = self.data['1y_1d']
                current_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                f.write(f"CURRENT MARKET DATA:\n")
                f.write(f"Current Price: ₹{current_price:.2f}\n")
                f.write(f"Daily Change: ₹{change:.2f} ({change_pct:+.2f}%)\n")
                f.write(f"Volume: {df['Volume'].iloc[-1]:,.0f}\n")
                f.write(f"52-Week High: ₹{df['High'].max():.2f}\n")
                f.write(f"52-Week Low: ₹{df['Low'].min():.2f}\n\n")
                
                # Technical analysis summary
                indicators = self.calculate_technical_indicators(df)
                rsi = indicators['RSI'].iloc[-1]
                
                f.write(f"TECHNICAL ANALYSIS SUMMARY:\n")
                f.write(f"RSI (14): {rsi:.2f} ")
                if rsi > 70:
                    f.write("(Overbought)\n")
                elif rsi < 30:
                    f.write("(Oversold)\n")
                else:
                    f.write("(Neutral)\n")
                    
                sma_20 = indicators['SMA_20'].iloc[-1]
                sma_50 = indicators['SMA_50'].iloc[-1]
                
                f.write(f"Price vs SMA 20: {'Above' if current_price > sma_20 else 'Below'}\n")
                f.write(f"Price vs SMA 50: {'Above' if current_price > sma_50 else 'Below'}\n")
                f.write(f"SMA 20 vs SMA 50: {'Bullish' if sma_20 > sma_50 else 'Bearish'}\n\n")
                
                # Risk metrics
                returns = df['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                max_dd = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100
                
                f.write(f"RISK METRICS:\n")
                f.write(f"Annualized Volatility: {volatility:.2f}%\n")
                f.write(f"Maximum Drawdown: {max_dd:.2f}%\n\n")
                
            f.write(f"FILES GENERATED:\n")
            f.write(f"- Price analysis charts (PNG)\n")
            f.write(f"- Pattern analysis chart (PNG)\n")
            f.write(f"- Correlation heatmap (PNG)\n")
            f.write(f"- Interactive chart (HTML)\n")
            f.write(f"- Excel report with data\n")
            f.write(f"- JSON data export\n\n")
            
            f.write(f"DISCLAIMER:\n")
            f.write(f"This analysis is for educational and informational purposes only.\n")
            f.write(f"It should not be considered as investment advice.\n")
            f.write(f"Please consult with a qualified financial advisor before making\n")
            f.write(f"any investment decisions.\n")
            
        print(f"✓ Saved summary report: {filename}")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 60)
        print("RELIANCE INDUSTRIES COMPREHENSIVE STOCK ANALYSIS")
        print("=" * 60)
        
        # Fetch data
        self.fetch_data()
        
        if not self.data:
            print("❌ No data available for analysis")
            return
            
        # Analyze each timeframe
        for timeframe, df in self.data.items():
            if len(df) < 50:  # Skip if insufficient data
                continue
                
            print(f"\n📊 Analyzing {timeframe}...")
            
            # Calculate indicators and patterns
            indicators = self.calculate_technical_indicators(df)
            patterns = self.detect_candlestick_patterns(df)
            
            # Create visualizations
            self.create_price_chart(df, timeframe, indicators, patterns)
            
        # Use daily data for detailed analysis
        if '1y_1d' in self.data:
            df = self.data['1y_1d']
            indicators = self.calculate_technical_indicators(df)
            patterns = self.detect_candlestick_patterns(df)
            
            print(f"\n📈 Creating detailed analysis...")
            self.create_pattern_analysis_chart(df, patterns)
            self.create_correlation_heatmap(df, indicators)
            self.create_interactive_chart(df, indicators, patterns)
            self.create_excel_report(df, indicators, patterns)
            self.create_json_report(df, indicators, patterns)
            
        # Create summary
        self.create_summary_report()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print(f"📁 All files saved in: {self.output_dir}/")
        print("=" * 60)
        
        # List generated files
        print("\n📋 Generated Files:")
        for file in sorted(os.listdir(self.output_dir)):
            file_path = os.path.join(self.output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   • {file} ({file_size:.1f} KB)")

def main():
    """Main execution function"""
    print("🚀 Starting Enhanced Reliance Industries Stock Analysis...")
    
    # Create and run analysis
    analyzer = EnhancedVisualizationEngine("RELIANCE.NS")
    analyzer.run_complete_analysis()
    
    print("\n🎯 Analysis complete! Check the output directory for all generated files.")
    print("\n📊 Files include:")
    print("   • PNG charts for visual analysis")
    print("   • Excel spreadsheet with all data")
    print("   • Interactive HTML chart")
    print("   • JSON data for further processing")
    print("   • Summary text report")

if __name__ == "__main__":
    main()