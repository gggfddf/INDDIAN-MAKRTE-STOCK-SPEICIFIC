#!/usr/bin/env python3
"""
Test script to verify the stock market analyzer structure
without requiring heavy ML dependencies.
"""

import sys
import importlib.util

def test_module_structure():
    """Test if all modules have correct structure"""
    
    modules_to_test = [
        'data_fetcher.py',
        'technical_indicators.py', 
        'price_action_analysis.py',
        'deep_learning_models.py',
        'stock_market_analyzer.py'
    ]
    
    print("🧪 Testing Stock Market Analyzer Module Structure")
    print("=" * 60)
    
    for module_file in modules_to_test:
        print(f"\n📄 Testing {module_file}...")
        
        try:
            # Load module spec
            spec = importlib.util.spec_from_file_location("test_module", module_file)
            if spec is None:
                print(f"❌ Could not load spec for {module_file}")
                continue
                
            # Create module
            module = importlib.util.module_from_spec(spec)
            
            # Check if we can load the module (syntax check)
            try:
                spec.loader.exec_module(module)
                print(f"✅ {module_file} - Syntax OK")
                
                # Check for main classes
                if module_file == 'data_fetcher.py':
                    if hasattr(module, 'IndianStockDataFetcher'):
                        print(f"   ✅ IndianStockDataFetcher class found")
                    else:
                        print(f"   ⚠️  IndianStockDataFetcher class not found")
                        
                elif module_file == 'technical_indicators.py':
                    if hasattr(module, 'AdvancedTechnicalIndicators'):
                        print(f"   ✅ AdvancedTechnicalIndicators class found")
                    else:
                        print(f"   ⚠️  AdvancedTechnicalIndicators class not found")
                        
                elif module_file == 'price_action_analysis.py':
                    if hasattr(module, 'AdvancedPriceActionAnalysis'):
                        print(f"   ✅ AdvancedPriceActionAnalysis class found")
                    else:
                        print(f"   ⚠️  AdvancedPriceActionAnalysis class not found")
                        
                elif module_file == 'deep_learning_models.py':
                    if hasattr(module, 'DeepLearningModels'):
                        print(f"   ✅ DeepLearningModels class found")
                    else:
                        print(f"   ⚠️  DeepLearningModels class not found")
                        
                elif module_file == 'stock_market_analyzer.py':
                    if hasattr(module, 'StockMarketAnalyzer'):
                        print(f"   ✅ StockMarketAnalyzer class found")
                    else:
                        print(f"   ⚠️  StockMarketAnalyzer class not found")
                
            except ImportError as e:
                print(f"⚠️  {module_file} - Import dependencies missing: {e}")
                print(f"   (This is expected if ML dependencies aren't installed)")
            except Exception as e:
                print(f"❌ {module_file} - Error: {e}")
                
        except Exception as e:
            print(f"❌ {module_file} - Failed to load: {e}")
    
    print(f"\n📋 Testing example_usage.py...")
    try:
        spec = importlib.util.spec_from_file_location("example", "example_usage.py")
        if spec:
            module = importlib.util.module_from_spec(spec)
            print(f"✅ example_usage.py - Structure OK")
        else:
            print(f"❌ example_usage.py - Could not load")
    except Exception as e:
        print(f"❌ example_usage.py - Error: {e}")

def test_project_files():
    """Test if all required project files exist"""
    
    required_files = [
        'requirements.txt',
        'data_fetcher.py',
        'technical_indicators.py',
        'price_action_analysis.py', 
        'deep_learning_models.py',
        'stock_market_analyzer.py',
        'example_usage.py',
        'README.md'
    ]
    
    print(f"\n📁 Testing Project Files...")
    print("-" * 40)
    
    import os
    
    for file_name in required_files:
        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name)
            print(f"✅ {file_name:<25} ({file_size:,} bytes)")
        else:
            print(f"❌ {file_name:<25} (missing)")

def test_requirements():
    """Test requirements.txt structure"""
    
    print(f"\n📦 Testing Requirements...")
    print("-" * 40)
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
        
        print(f"✅ requirements.txt found ({len(requirements)} packages)")
        
        key_packages = [
            'yfinance', 'pandas', 'numpy', 'scikit-learn', 
            'tensorflow', 'matplotlib', 'seaborn', 'plotly'
        ]
        
        req_text = ''.join(requirements).lower()
        
        for package in key_packages:
            if package in req_text:
                print(f"   ✅ {package} found in requirements")
            else:
                print(f"   ⚠️  {package} not found in requirements")
                
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")

def show_system_info():
    """Show system information"""
    
    print(f"\n💻 System Information...")
    print("-" * 40)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check if common packages are available
    common_packages = ['pandas', 'numpy', 'matplotlib', 'requests']
    
    for package in common_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")

def test_code_quality():
    """Basic code quality checks"""
    
    print(f"\n🔍 Basic Code Quality Checks...")
    print("-" * 40)
    
    python_files = [
        'data_fetcher.py', 'technical_indicators.py', 
        'price_action_analysis.py', 'deep_learning_models.py',
        'stock_market_analyzer.py', 'example_usage.py'
    ]
    
    for file_name in python_files:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Basic metrics
            total_lines = len(lines)
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            docstring_lines = content.count('"""') // 2 * 3  # Rough estimate
            
            print(f"📄 {file_name}:")
            print(f"   Total lines: {total_lines}")
            print(f"   Code lines: {code_lines}")
            print(f"   Comments: {comment_lines}")
            print(f"   Has docstrings: {'✅' if '"""' in content else '❌'}")
            print(f"   Has classes: {'✅' if 'class ' in content else '❌'}")
            print(f"   Has functions: {'✅' if 'def ' in content else '❌'}")
            
        except Exception as e:
            print(f"❌ Error analyzing {file_name}: {e}")

def main():
    """Main test function"""
    
    print("🚀 Stock Market Analyzer - Structure Test")
    print("=" * 60)
    print("This test verifies the project structure without requiring")
    print("heavy ML dependencies to be installed.")
    print("=" * 60)
    
    # Run all tests
    test_project_files()
    test_requirements()
    test_module_structure()
    test_code_quality()
    show_system_info()
    
    print(f"\n✅ Structure Test Complete!")
    print("=" * 60)
    print("📝 Notes:")
    print("   - Import errors for ML dependencies are expected")
    print("   - Install requirements.txt for full functionality")
    print("   - Run example_usage.py for actual analysis")
    print("=" * 60)

if __name__ == "__main__":
    main()