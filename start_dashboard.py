#!/usr/bin/env python3
"""
Startup script for the E-commerce Analytics Dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Start the FastAPI dashboard server"""
    print("🚀 Starting E-commerce Analytics Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8080")
    print("📁 Make sure you have run the analysis first: python run_analysis.py")
    print("-" * 60)
    
    try:
        # Start the FastAPI server
        subprocess.run([
            sys.executable, "api_server.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
