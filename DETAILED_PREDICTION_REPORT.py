#!/usr/bin/env python3
"""
DETAILED PREDICTION REPORT GENERATOR
====================================

Generates comprehensive prediction reports for each timeframe separately
with movement analysis, predicted values, confidence, and probability.
"""

import json
import pandas as pd
from datetime import datetime
import os

def generate_detailed_prediction_report():
    """Generate detailed prediction report for all timeframes"""
    
    # Load the price action analysis report (contains ML predictions)
    try:
        with open('nsei_final_demo/reports/price_action_analysis_report.json', 'r') as f:
            price_action_data = json.load(f)
    except FileNotFoundError:
        print("❌ Price action report not found. Please run the analysis first.")
        return
    
    ml_predictions = price_action_data.get('ml_predictions', {})
    
    if not ml_predictions:
        print("❌ No ML predictions found in the report.")
        return
    
    print("=" * 100)
    print("📊 COMPREHENSIVE PREDICTION REPORT - NIFTY 50 INDEX")
    print("=" * 100)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbol: ^NSEI (NIFTY 50 Index)")
    print(f"Total Timeframes Analyzed: {len(ml_predictions)}")
    print("=" * 100)
    
    # Generate detailed report for each timeframe
    detailed_report = {
        'report_title': 'COMPREHENSIVE PREDICTION REPORT - ALL TIMEFRAMES',
        'symbol': '^NSEI',
        'company_name': 'NIFTY 50 Index',
        'analysis_timestamp': datetime.now().isoformat(),
        'timeframe_predictions': {}
    }
    
    timeframe_order = ['5m', '15m', '1h', '1d', '1wk']
    
    for tf_key in timeframe_order:
        if tf_key in ml_predictions:
            print(f"\n{'🔍 ' + ml_predictions[tf_key]['timeframe'].upper() + ' TIMEFRAME PREDICTION':=^100}")
            
            pred = ml_predictions[tf_key]
            
            # Extract prediction details
            direction = pred['direction_prediction']
            confidence = pred['confidence']
            price_targets = pred['price_targets']
            time_horizons = pred['time_horizons']
            risk = pred['risk_assessment']
            specific_pred = pred['specific_prediction']
            
            current_price = price_targets['current_price']
            upside_target = price_targets['upside_target']
            downside_target = price_targets['downside_target']
            upside_prob = price_targets['upside_probability']
            downside_prob = price_targets['downside_probability']
            conf_lower = price_targets['confidence_interval_lower']
            conf_upper = price_targets['confidence_interval_upper']
            expected_move = price_targets['expected_move_percent']
            
            print(f"\n📈 PRICE MOVEMENT ANALYSIS:")
            print(f"   Current Price:           ₹{current_price:,.2f}")
            print(f"   Direction Prediction:    {direction}")
            print(f"   Confidence Level:        {confidence:.1%}")
            print(f"   Risk Assessment:         {risk}")
            
            print(f"\n🎯 SPECIFIC PRICE TARGETS:")
            print(f"   Upside Target:           ₹{upside_target:,.2f}")
            print(f"   Downside Target:         ₹{downside_target:,.2f}")
            print(f"   Expected Move:           {expected_move:.2f}%")
            
            print(f"\n📊 PROBABILITY ANALYSIS:")
            print(f"   Upside Probability:      {upside_prob:.1%}")
            print(f"   Downside Probability:    {downside_prob:.1%}")
            
            print(f"\n🔒 CONFIDENCE INTERVALS (95%):")
            print(f"   Lower Bound:             ₹{conf_lower:,.2f}")
            print(f"   Upper Bound:             ₹{conf_upper:,.2f}")
            print(f"   Confidence Range:        ₹{conf_upper - conf_lower:,.2f}")
            
            print(f"\n⏰ TIME HORIZONS:")
            print(f"   Short Term:              {time_horizons['short']}")
            print(f"   Medium Term:             {time_horizons['medium']}")
            print(f"   Long Term:               {time_horizons['long']}")
            
            print(f"\n🎲 SPECIFIC PREDICTION:")
            print(f"   {specific_pred}")
            
            # Calculate movement percentages
            upside_move = ((upside_target - current_price) / current_price) * 100
            downside_move = ((current_price - downside_target) / current_price) * 100
            
            print(f"\n📈 MOVEMENT CALCULATIONS:")
            print(f"   Upside Movement:         +{upside_move:.2f}%")
            print(f"   Downside Movement:       -{downside_move:.2f}%")
            
            # Feature importance
            if 'feature_importance' in pred:
                print(f"\n🧠 FEATURE IMPORTANCE:")
                for feature, importance in pred['feature_importance'].items():
                    print(f"   {feature.replace('_', ' ').title():<20} {importance:.1%}")
            
            # Add to detailed report
            detailed_report['timeframe_predictions'][tf_key] = {
                'timeframe_name': pred['timeframe'],
                'direction_prediction': direction,
                'confidence_percentage': f"{confidence:.1%}",
                'risk_level': risk,
                'current_price_inr': f"₹{current_price:,.2f}",
                'upside_target_inr': f"₹{upside_target:,.2f}",
                'downside_target_inr': f"₹{downside_target:,.2f}",
                'upside_movement_percent': f"+{upside_move:.2f}%",
                'downside_movement_percent': f"-{downside_move:.2f}%",
                'upside_probability': f"{upside_prob:.1%}",
                'downside_probability': f"{downside_prob:.1%}",
                'confidence_interval_lower': f"₹{conf_lower:,.2f}",
                'confidence_interval_upper': f"₹{conf_upper:,.2f}",
                'confidence_range': f"₹{conf_upper - conf_lower:,.2f}",
                'expected_move_percent': f"{expected_move:.2f}%",
                'time_horizons': time_horizons,
                'specific_prediction_text': specific_pred,
                'feature_importance': pred.get('feature_importance', {}),
                'volatility_percentile': pred.get('volatility_percentile', 0),
                'volume_trend': pred.get('volume_trend', 1.0)
            }
            
            print("-" * 100)
    
    # Generate summary analysis
    print(f"\n{'📊 CROSS-TIMEFRAME SUMMARY ANALYSIS':=^100}")
    
    # Collect all predictions for summary
    all_directions = []
    all_confidences = []
    all_risks = []
    
    for tf_pred in detailed_report['timeframe_predictions'].values():
        all_directions.append(tf_pred['direction_prediction'])
        all_confidences.append(float(tf_pred['confidence_percentage'].rstrip('%'))/100)
        all_risks.append(tf_pred['risk_level'])
    
    # Summary statistics
    bullish_count = all_directions.count('BULLISH')
    bearish_count = all_directions.count('BEARISH')
    neutral_count = all_directions.count('NEUTRAL')
    avg_confidence = sum(all_confidences) / len(all_confidences)
    
    print(f"\n🎯 OVERALL MARKET SENTIMENT:")
    print(f"   Bullish Timeframes:      {bullish_count}/{len(all_directions)}")
    print(f"   Bearish Timeframes:      {bearish_count}/{len(all_directions)}")
    print(f"   Neutral Timeframes:      {neutral_count}/{len(all_directions)}")
    print(f"   Average Confidence:      {avg_confidence:.1%}")
    
    # Dominant signal
    if bullish_count > bearish_count and bullish_count > neutral_count:
        dominant = "BULLISH"
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        dominant = "BEARISH"
    else:
        dominant = "NEUTRAL/MIXED"
    
    print(f"   Dominant Signal:         {dominant}")
    
    # Risk distribution
    high_risk = all_risks.count('HIGH')
    medium_risk = all_risks.count('MEDIUM')
    low_risk = all_risks.count('LOW')
    
    print(f"\n⚠️ RISK DISTRIBUTION:")
    print(f"   High Risk Timeframes:    {high_risk}/{len(all_risks)}")
    print(f"   Medium Risk Timeframes:  {medium_risk}/{len(all_risks)}")
    print(f"   Low Risk Timeframes:     {low_risk}/{len(all_risks)}")
    
    # Save detailed prediction report
    report_filename = 'nsei_final_demo/reports/DETAILED_PREDICTION_REPORT.json'
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)
    
    with open(report_filename, 'w') as f:
        json.dump(detailed_report, f, indent=2, default=str)
    
    print(f"\n💾 DETAILED PREDICTION REPORT SAVED:")
    print(f"   File: {report_filename}")
    print(f"   Size: {os.path.getsize(report_filename)} bytes")
    
    print("\n" + "=" * 100)
    print("✅ DETAILED PREDICTION REPORT GENERATION COMPLETE")
    print("=" * 100)
    
    return detailed_report

def create_prediction_summary_table():
    """Create a summary table of all predictions"""
    
    try:
        with open('nsei_final_demo/reports/price_action_analysis_report.json', 'r') as f:
            price_action_data = json.load(f)
    except FileNotFoundError:
        print("❌ Price action report not found.")
        return
    
    ml_predictions = price_action_data.get('ml_predictions', {})
    
    print(f"\n{'📊 PREDICTION SUMMARY TABLE':=^120}")
    print("-" * 120)
    print(f"{'Timeframe':<12} {'Direction':<10} {'Confidence':<12} {'Current Price':<15} {'Target Price':<15} {'Movement':<12} {'Risk':<8} {'Time Horizon'}")
    print("-" * 120)
    
    timeframe_order = ['5m', '15m', '1h', '1d', '1wk']
    
    for tf_key in timeframe_order:
        if tf_key in ml_predictions:
            pred = ml_predictions[tf_key]
            
            direction = pred['direction_prediction']
            confidence = f"{pred['confidence']:.1%}"
            current_price = f"₹{pred['price_targets']['current_price']:,.0f}"
            
            if direction == 'BULLISH':
                target_price = f"₹{pred['price_targets']['upside_target']:,.0f}"
                movement = f"+{pred['price_targets']['expected_move_percent']:.1f}%"
            elif direction == 'BEARISH':
                target_price = f"₹{pred['price_targets']['downside_target']:,.0f}"
                movement = f"-{pred['price_targets']['expected_move_percent']:.1f}%"
            else:
                target_price = current_price
                movement = f"±{pred['price_targets']['expected_move_percent']:.1f}%"
            
            risk = pred['risk_assessment']
            time_horizon = pred['time_horizons']['short']
            
            print(f"{pred['timeframe']:<12} {direction:<10} {confidence:<12} {current_price:<15} {target_price:<15} {movement:<12} {risk:<8} {time_horizon}")
    
    print("-" * 120)

if __name__ == "__main__":
    print("🚀 Starting Detailed Prediction Report Generation...")
    
    # Generate comprehensive prediction report
    detailed_report = generate_detailed_prediction_report()
    
    # Generate summary table
    create_prediction_summary_table()
    
    print("\n🎉 All prediction reports generated successfully!")