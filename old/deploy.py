#!/usr/bin/env python3
"""
Deployment helper script for Ecommerce Analysis Dashboard
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if all required files exist for deployment."""
    required_files = [
        'requirements.txt',
        'Procfile',
        'src/api/fastapi_service.py',
        'src/dashboard/index.html',
        'datasets/processed/sales_data_jan_june_2025.csv',
        'datasets/processed/returns_jan_june_2025.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files for deployment:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present for deployment")
    return True

def test_local():
    """Test if the app runs locally."""
    print("🧪 Testing local deployment...")
    print("ℹ️  Note: Make sure virtual environment is activated (source .venv/bin/activate)")
    try:
        # Test if we can import the FastAPI app
        import sys
        sys.path.append('.')
        from src.api.fastapi_service import app
        print("✅ FastAPI app imports successfully")
        
        # Test if data files are accessible
        from src.api.fastapi_service import load_sales_data, load_returns_data
        sales_data = load_sales_data()
        returns_data = load_returns_data()
        print(f"✅ Data files loaded: {len(sales_data)} sales records, {len(returns_data)} returns records")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to activate virtual environment: source .venv/bin/activate")
        return False
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

def main():
    """Main deployment helper."""
    print("🚀 Ecommerce Dashboard Deployment Helper")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Deployment requirements not met. Please fix missing files.")
        return
    
    # Test locally
    if not test_local():
        print("\n❌ Local test failed. Please fix issues before deploying.")
        return
    
    print("\n✅ Ready for deployment!")
    print("\n📋 Next steps:")
    print("1. Commit all changes: git add . && git commit -m 'Deploy to production'")
    print("2. Push to GitHub: git push origin main")
    print("3. Deploy to Railway/Render/Heroku using the instructions in DEPLOYMENT.md")
    print("\n🌐 Your dashboard will be available at the provided URL")

if __name__ == "__main__":
    main()
