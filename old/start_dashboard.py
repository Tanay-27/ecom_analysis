#!/usr/bin/env python3
"""
Quick start script for the Ecommerce Analysis Dashboard.
"""

import subprocess
import sys
import os

def main():
    """Start the dashboard."""
    print("🚀 Starting Ecommerce Analysis Dashboard...")
    print("=" * 50)
    
    try:
        print("🌐 Starting FastAPI server on http://localhost:8000")
        print("📊 Dashboard will be available at http://localhost:8000")
        print("=" * 50)
        
        # Set PYTHONPATH and start the FastAPI service
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'
        
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api.fastapi_service:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], env=env)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
