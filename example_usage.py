#!/usr/bin/env python3
"""
Example Usage of Advanced Stock Market Analyzer

This script demonstrates how to use the comprehensive deep learning
stock market analysis module for Indian markets.

Features demonstrated:
- Data fetching for multiple timeframes
- Technical indicator analysis (40+ indicators)
- Price action analysis with ML pattern discovery
- Deep learning predictions (LSTM, CNN, Attention models)
- Comprehensive report generation
- HTML export functionality

Usage:
    python example_usage.py [SYMBOL]
    
Example:
    python example_usage.py RELIANCE.NS
    python example_usage.py TCS.NS
    python example_usage.py INFY.NS
"""

import sys
import json
from datetime import datetime
from stock_market_analyzer import StockMarketAnalyzer

def analyze_stock(symbol="RELIANCE.NS", enable_ml=False):
    """
    Analyze a stock with comprehensive technical and price action analysis
    
    Args:
        symbol (str): Stock symbol to analyze
        enable_ml (bool): Whether to enable ML model training (resource intensive)
    """
    
    print("🚀 Advanced Stock Market Analysis System")
    print("=" * 60)
    print(f"📊 Analyzing: {symbol}")
    print(f"🤖 ML Training: {'Enabled' if enable_ml else 'Disabled (for demo)'}")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = StockMarketAnalyzer(symbol)
        
        # Step 1: Fetch comprehensive data
        print("\n📡 Step 1: Fetching Market Data")
        print("-" * 40)
        data = analyzer.fetch_comprehensive_data()
        
        if not data:
            print("❌ No data available for this symbol")
            return None
        
        # Step 2: Run comprehensive analysis
        print("\n🔍 Step 2: Running Comprehensive Analysis")
        print("-" * 40)
        analysis_success = analyzer.run_comprehensive_analysis(train_models=enable_ml)
        
        if not analysis_success:
            print("❌ Analysis failed")
            return None
        
        # Step 3: Generate technical analysis report
        print("\n📈 Step 3: Generating Technical Analysis Report")
        print("-" * 40)
        technical_report = analyzer.generate_technical_analysis_report()
        
        # Step 4: Generate price action report
        print("\n🕯️ Step 4: Generating Price Action Report")
        print("-" * 40)
        price_action_report = analyzer.generate_price_action_report()
        
        # Step 5: Display key insights
        print("\n💡 Step 5: Key Market Insights")
        print("-" * 40)
        display_key_insights(technical_report, price_action_report)
        
        # Step 6: Export reports
        print("\n📊 Step 6: Exporting Analysis Reports")
        print("-" * 40)
        html_file = analyzer.export_analysis_to_html(
            technical_report, 
            price_action_report
        )
        
        # Save JSON reports for programmatic access
        json_files = save_json_reports(technical_report, price_action_report, symbol)
        
        print(f"\n✅ Analysis Complete!")
        print("=" * 60)
        print(f"📄 HTML Report: {html_file}")
        print(f"📁 JSON Reports: {', '.join(json_files)}")
        print("=" * 60)
        
        return {
            'technical_report': technical_report,
            'price_action_report': price_action_report,
            'html_file': html_file,
            'json_files': json_files
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def display_key_insights(technical_report, price_action_report):
    """Display key insights from the analysis"""
    
    # Technical Analysis Insights
    print("📈 TECHNICAL ANALYSIS INSIGHTS")
    print("   " + "─" * 35)
    
    overall_sentiment = technical_report.get('overall_sentiment', {})
    sentiment = overall_sentiment.get('overall_sentiment', 'Neutral')
    sentiment_score = overall_sentiment.get('sentiment_score', 0)
    
    # Color code sentiment
    if sentiment == 'Bullish':
        sentiment_display = f"🟢 {sentiment} ({sentiment_score:.2f})"
    elif sentiment == 'Bearish':
        sentiment_display = f"🔴 {sentiment} ({sentiment_score:.2f})"
    else:
        sentiment_display = f"🟡 {sentiment} ({sentiment_score:.2f})"
    
    print(f"   Overall Sentiment: {sentiment_display}")
    
    # Risk Assessment
    risk_assessment = technical_report.get('risk_assessment', {})
    risk_level = risk_assessment.get('risk_level', 'Unknown')
    risk_score = risk_assessment.get('risk_score', 0)
    
    if risk_level == 'High':
        risk_display = f"🔴 {risk_level} ({risk_score:.2f})"
    elif risk_level == 'Medium':
        risk_display = f"🟡 {risk_level} ({risk_score:.2f})"
    else:
        risk_display = f"🟢 {risk_level} ({risk_score:.2f})"
    
    print(f"   Risk Level: {risk_display}")
    
    # Trading Signals
    trading_signals = technical_report.get('trading_signals', {})
    print(f"   Trading Signals:")
    for timeframe, signal_data in trading_signals.items():
        signal = signal_data.get('signal', 'HOLD')
        strength = signal_data.get('strength', 0)
        confidence = signal_data.get('confidence', 0)
        
        if signal == 'BUY':
            signal_display = f"🟢 {signal}"
        elif signal == 'SELL':
            signal_display = f"🔴 {signal}"
        else:
            signal_display = f"🟡 {signal}"
        
        print(f"     {timeframe.title()}: {signal_display} (Strength: {strength:.2f}, Confidence: {confidence:.2f})")
    
    # Price Action Insights
    print("\n🕯️ PRICE ACTION ANALYSIS INSIGHTS")
    print("   " + "─" * 38)
    
    # Pattern Reliability
    pattern_reliability = price_action_report.get('pattern_reliability', {})
    reliability_level = pattern_reliability.get('reliability_level', 'Unknown')
    overall_reliability = pattern_reliability.get('overall_reliability', 0)
    
    if reliability_level == 'High':
        reliability_display = f"🟢 {reliability_level} ({overall_reliability:.2f})"
    elif reliability_level == 'Medium':
        reliability_display = f"🟡 {reliability_level} ({overall_reliability:.2f})"
    else:
        reliability_display = f"🔴 {reliability_level} ({overall_reliability:.2f})"
    
    print(f"   Pattern Reliability: {reliability_display}")
    
    # Candlestick Patterns Summary
    candlestick_analysis = price_action_report.get('candlestick_analysis', {})
    total_patterns = 0
    total_bullish = 0
    total_bearish = 0
    
    for timeframe, data in candlestick_analysis.items():
        total_patterns += data.get('total_patterns_detected', 0)
        total_bullish += data.get('bullish_patterns', 0)
        total_bearish += data.get('bearish_patterns', 0)
    
    print(f"   Candlestick Patterns: {total_patterns} detected")
    print(f"     🟢 Bullish: {total_bullish}")
    print(f"     🔴 Bearish: {total_bearish}")
    
    # Breakout Analysis
    breakout_analysis = price_action_report.get('breakout_analysis', {})
    if breakout_analysis:
        print(f"   Breakout Signals: {len(breakout_analysis)} timeframes")
        for timeframe, data in breakout_analysis.items():
            direction = data.get('direction', 'Unknown')
            strength = data.get('strength', 0)
            if direction != 'Unknown':
                direction_display = f"🟢 {direction}" if direction == 'Upward' else f"🔴 {direction}"
                print(f"     {timeframe}: {direction_display} (Strength: {strength:.2f})")
    
    # ML Insights (if available)
    ml_insights = technical_report.get('ml_insights', {})
    if ml_insights:
        print("\n🤖 MACHINE LEARNING INSIGHTS")
        print("   " + "─" * 32)
        
        prediction_count = 0
        for key, value in ml_insights.items():
            if 'prediction' in key and isinstance(value, dict):
                prediction_count += 1
                timeframe = key.replace('_prediction', '')
                direction = value.get('predicted_direction', 'Unknown')
                change_pct = value.get('price_change_percent', 0)
                confidence = value.get('confidence', 0)
                
                direction_display = f"🟢 {direction}" if direction == 'Up' else f"🔴 {direction}"
                print(f"   {timeframe}: {direction_display} ({change_pct:+.2f}%, Confidence: {confidence:.2f})")
        
        if prediction_count == 0:
            print("   No ML predictions available (enable ML training for predictions)")

def save_json_reports(technical_report, price_action_report, symbol):
    """Save reports as JSON files for programmatic access"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Technical analysis JSON
    tech_filename = f"{symbol}_technical_analysis_{timestamp}.json"
    with open(tech_filename, 'w', encoding='utf-8') as f:
        json.dump(technical_report, f, indent=2, default=str)
    
    # Price action JSON
    pa_filename = f"{symbol}_price_action_{timestamp}.json"
    with open(pa_filename, 'w', encoding='utf-8') as f:
        json.dump(price_action_report, f, indent=2, default=str)
    
    return [tech_filename, pa_filename]

def get_popular_indian_stocks():
    """Get list of popular Indian stocks for analysis"""
    return {
        'RELIANCE.NS': 'Reliance Industries Ltd.',
        'TCS.NS': 'Tata Consultancy Services Ltd.',
        'INFY.NS': 'Infosys Ltd.',
        'HDFCBANK.NS': 'HDFC Bank Ltd.',
        'ICICIBANK.NS': 'ICICI Bank Ltd.',
        'HINDUNILVR.NS': 'Hindustan Unilever Ltd.',
        'ITC.NS': 'ITC Ltd.',
        'SBIN.NS': 'State Bank of India',
        'BHARTIARTL.NS': 'Bharti Airtel Ltd.',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank Ltd.',
        'LT.NS': 'Larsen & Toubro Ltd.',
        'ASIANPAINT.NS': 'Asian Paints Ltd.',
        'MARUTI.NS': 'Maruti Suzuki India Ltd.',
        'TITAN.NS': 'Titan Company Ltd.',
        'WIPRO.NS': 'Wipro Ltd.'
    }

def main():
    """Main function"""
    
    # Get symbol from command line or use default
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        if not symbol.endswith('.NS'):
            symbol += '.NS'
    else:
        symbol = "RELIANCE.NS"
    
    # Check if ML training should be enabled
    enable_ml = '--ml' in sys.argv or '-m' in sys.argv
    
    if enable_ml:
        print("⚠️  ML training enabled - this will take significantly longer but provide predictions")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            enable_ml = False
            print("Proceeding without ML training...")
    
    # Show available stocks if requested
    if '--list' in sys.argv or '-l' in sys.argv:
        print("\n📋 Popular Indian Stocks Available for Analysis:")
        print("=" * 60)
        stocks = get_popular_indian_stocks()
        for symbol, name in stocks.items():
            print(f"   {symbol:<15} - {name}")
        print("\nUsage: python example_usage.py [SYMBOL] [--ml]")
        return
    
    # Run analysis
    result = analyze_stock(symbol, enable_ml)
    
    if result:
        print(f"\n💡 Tip: Use '{symbol}' in your trading platform")
        print("⚠️  Disclaimer: This analysis is for educational purposes only.")
        print("   Always consult with financial advisors before making investment decisions.")
    
    return result

if __name__ == "__main__":
    # Handle help request
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    main()