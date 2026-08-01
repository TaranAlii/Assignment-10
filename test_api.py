import subprocess
import time
import urllib.request
import json
import os
import sys

def main():
    print("Launching Flask API server in the background...")
    # Change dir to assignment-10 to make sure relative paths load model.pkl properly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Start the Flask app
    proc = subprocess.Popen(["../.venv/bin/python", "app.py"])
    
    # Wait for the Flask server to boot up
    time.sleep(3)
    
    try:
        url = "http://127.0.0.1:5000/predict"
        
        # Test case: patient details
        payload = {
            "age": 52.0,
            "sex": 1.0,
            "cp": 0.0,
            "trestbps": 125.0,
            "chol": 212.0,
            "fbs": 0.0,
            "restecg": 1.0,
            "thalach": 168.0,
            "exang": 0.0,
            "oldpeak": 1.0,
            "slope": 2.0,
            "ca": 2.0,
            "thal": 3.0
        }
        
        print("Sending POST request to /predict with JSON payload...")
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = response.read().decode('utf-8')
            res = json.loads(res_data)
            print("API Response:")
            print(json.dumps(res, indent=2))
            
            # Assertions
            prediction = res.get("prediction")
            print(f"Prediction result: '{prediction}'")
            if prediction in ["No Heart Disease Detected", "Heart Disease Detected"]:
                print("SUCCESS: API returned a valid prediction!")
            else:
                print("FAILURE: API returned unexpected prediction.")
                sys.exit(1)
                
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.reason)
        try:
            print("Response body:", e.read().decode('utf-8'))
        except:
            pass
        sys.exit(1)
    except Exception as e:
        print("Error during API request test:", e)
        sys.exit(1)
    finally:
        print("Stopping Flask background server...")
        proc.terminate()
        proc.wait()
        print("Server stopped.")

if __name__ == "__main__":
    main()
