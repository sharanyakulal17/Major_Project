import os
import sys

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projectssssssss")
sys.path.insert(0, TARGET_DIR)
os.chdir(TARGET_DIR)

# Execute the inner app.py with its full module scope
if __name__ == "__main__":
    import app
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Self-Healing Backend server on http://127.0.0.1:{port}...")
    app.app.run(host="0.0.0.0", port=port, debug=False)
