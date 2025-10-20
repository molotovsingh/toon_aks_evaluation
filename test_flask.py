#!/usr/bin/env python3
import subprocess
import time
import requests

print("🚀 Starting Flask app...")
process = subprocess.Popen(['uv', 'run', 'python', 'flask_app.py'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)

print("⏳ Waiting for startup...")
time.sleep(3)

try:
    # Test the endpoint
    response = requests.get('http://localhost:5001/test', timeout=5)
    print(f"✅ Response: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print("✅ Flask app working!")
    else:
        print(f"❌ Bad status: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")

finally:
    print("🛑 Stopping Flask...")
    process.terminate()
    process.wait()
    print("✅ Test complete")
