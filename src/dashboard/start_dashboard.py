#!/usr/bin/env python3
"""
Simple dashboard startup script for the organized project structure.
"""

import subprocess
import sys
import os

def main():
    """Start the FastAPI dashboard service."""
    print("🚀 Starting Ecommerce Analysis Dashboard...")
    print("=" * 50)
    
    try:
        # Change to the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_root)
        
        print(f"📁 Working directory: {os.getcwd()}")
        print("🌐 Starting FastAPI server on http://localhost:8000")
        print("📊 Dashboard will be available at http://localhost:8000")
        print("=" * 50)
        
        # Start the FastAPI service
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api.fastapi_service:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()