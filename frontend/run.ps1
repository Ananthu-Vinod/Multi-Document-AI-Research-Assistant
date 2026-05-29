# Run Streamlit UI (uses python -m — no PATH entry needed)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

python -m pip install -q -r requirements.txt
python -m streamlit run streamlit_app.py
