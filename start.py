"""Start the whole app with one command:  python start.py

Starts the FastAPI backend (port 8000) and the Streamlit interface (port 8501), waits until the
backend is healthy, then opens the browser. Press Ctrl+C to stop both.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

ROOT = Path(__file__).parent
API, UI = "http://localhost:8000", "http://localhost:8501"


def main() -> None:
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "guideline_rag.api:app", "--port", "8000"], cwd=ROOT)
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501",
                           "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT)
    try:
        for _ in range(60):  # wait up to ~2 minutes for the backend to load
            try:
                health = requests.get(f"{API}/health", timeout=5).json()
                print("Backend health:", health)
                if not health.get("ok"):
                    print("Something is missing (is Ollama running? did you run "
                          "`python -m guideline_rag.index --rebuild`?)")
                break
            except requests.RequestException:
                time.sleep(2)
        webbrowser.open(UI)
        print(f"\nOpen {UI}   (API docs: {API}/docs)   Press Ctrl+C to stop.")
        api.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            p.terminate()


if __name__ == "__main__":
    main()
