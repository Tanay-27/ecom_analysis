#!/usr/bin/env python3
"""
Simple dashboard runner that works with the new project structure.
"""

import subprocess
import sys
import os

def main():
    """Start the dashboard."""
    print("🚀 Starting Ecommerce Analysis Dashboard...")
    print("=" * 50)
    
    try:
        # Add the current directory to Python path
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        print("🌐 Starting FastAPI server on http://localhost:8000")
        print("📊 Dashboard will be available at http://localhost:8000")
        print("=" * 50)
        
        # Import and run the FastAPI app directly
        from src.api.fastapi_service import app
        import uvicorn
        
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
