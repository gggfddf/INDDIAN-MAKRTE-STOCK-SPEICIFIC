#!/usr/bin/env python3
"""
Display Analysis Results
========================

This script displays the key findings from the Reliance Industries analysis
in an easy-to-understand format.
"""

import json
import os
from datetime import datetime

def display_analysis_results():
    """Display the key analysis results"""
    
    print("🏢" + "="*80)
    print("📊 RELIANCE INDUSTRIES LIMITED - COMPREHENSIVE ANALYSIS RESULTS")
    print("🏢" + "="*80)
    
    # Read JSON data
    json_file = "reliance_analysis_output/analysis_data.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n📈 CURRENT MARKET STATUS:")
        print(f"   Stock Symbol: {data['symbol']}")
        print(f"   Company: {data['company_name']}")
        print(f"   Current Price: ₹{data['current_price']:.2f}")
        print(f"   Daily Change: ₹{data['daily_change']:.2f} ({data['daily_change_percent']:.2f}%)")
        print(f"   Volume: {data['volume']:,}")
        
        print(f"\n🔬 TECHNICAL INDICATORS:")
        indicators = data['technical_indicators']
        rsi = indicators['rsi']
        print(f"   RSI (14): {rsi:.2f}", end="")
        if rsi > 70:
            print(" 🔴 (Overbought)")
        elif rsi < 30:
            print(" 🟢 (Oversold)")
        else:
            print(" 🟡 (Neutral)")
            
        print(f"   MACD: {indicators['macd']:.2f}")
        print(f"   SMA 20: ₹{indicators['sma_20']:.2f}")
        print(f"   SMA 50: ₹{indicators['sma_50']:.2f}")
        
        print(f"\n📊 CANDLESTICK PATTERNS DETECTED:")
        patterns = data['pattern_counts']
        for pattern, count in patterns.items():
            emoji = "🟢" if count > 20 else "🟡" if count > 10 else "🔴"
            print(f"   {emoji} {pattern.replace('_', ' ')}: {count} occurrences")
        
        print(f"\n⚠️  RISK METRICS:")
        risk = data['risk_metrics']
        print(f"   Volatility (Annual): {risk['volatility']:.2f}%")
        print(f"   Maximum Drawdown: {risk['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        
        # Risk assessment
        volatility = risk['volatility']
        if volatility < 15:
            risk_level = "🟢 LOW"
        elif volatility < 25:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🔴 HIGH"
        print(f"   Risk Level: {risk_level}")
        
    print(f"\n📁 GENERATED FILES:")
    output_dir = "reliance_analysis_output"
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        for file in files:
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            
            if file.endswith('.png'):
                emoji = "🖼️ "
            elif file.endswith('.html'):
                emoji = "🌐"
            elif file.endswith('.xlsx'):
                emoji = "📊"
            elif file.endswith('.json'):
                emoji = "📄"
            elif file.endswith('.txt'):
                emoji = "📝"
            else:
                emoji = "📎"
                
            print(f"   {emoji} {file} ({file_size:.1f} KB)")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Stock is currently trading at ₹{data['current_price']:.2f}")
    
    current_price = data['current_price']
    sma_20 = indicators['sma_20']
    sma_50 = indicators['sma_50']
    
    if current_price > sma_20 and current_price > sma_50:
        trend = "🟢 BULLISH"
    elif current_price < sma_20 and current_price < sma_50:
        trend = "🔴 BEARISH"
    else:
        trend = "🟡 MIXED"
    print(f"   • Current Trend: {trend}")
    
    if data['daily_change_percent'] > 1:
        momentum = "🚀 STRONG POSITIVE"
    elif data['daily_change_percent'] > 0:
        momentum = "🟢 POSITIVE"
    elif data['daily_change_percent'] > -1:
        momentum = "🟡 NEUTRAL"
    else:
        momentum = "🔴 NEGATIVE"
    print(f"   • Daily Momentum: {momentum}")
    
    # Volume analysis
    volume = data['volume']
    if volume > 15000000:
        volume_status = "🔥 HIGH"
    elif volume > 8000000:
        volume_status = "🟡 NORMAL"
    else:
        volume_status = "🔵 LOW"
    print(f"   • Volume Activity: {volume_status}")
    
    print(f"\n📋 RECOMMENDATION SUMMARY:")
    print(f"   • Technical Analysis: Based on RSI of {rsi:.1f}, the stock is in neutral territory")
    print(f"   • Pattern Analysis: {sum(patterns.values())} candlestick patterns detected")
    print(f"   • Risk Assessment: {risk_level.split()[1]} risk based on {volatility:.1f}% volatility")
    
    print(f"\n⚠️  IMPORTANT DISCLAIMERS:")
    print(f"   • This analysis is for educational purposes only")
    print(f"   • Not financial advice - consult a professional advisor")
    print(f"   • Past performance does not guarantee future results")
    print(f"   • Consider your risk tolerance before making decisions")
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🏢" + "="*80)

if __name__ == "__main__":
    display_analysis_results()