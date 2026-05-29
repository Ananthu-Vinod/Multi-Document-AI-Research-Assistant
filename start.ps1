# Start FastAPI backend + Streamlit frontend (Windows)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Starting backend on http://localhost:8000 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$Root\backend'; if (Test-Path .venv\Scripts\Activate.ps1) { .\.venv\Scripts\Activate.ps1 }; uvicorn app:app --host 0.0.0.0 --port 8000"
)

Start-Sleep -Seconds 4

Write-Host "Starting Streamlit on http://localhost:8501 ..."
Set-Location "$Root\frontend"
if (Test-Path .venv\Scripts\Activate.ps1) { .\.venv\Scripts\Activate.ps1 }
python -m pip install -q -r requirements.txt
python -m streamlit run streamlit_app.py
