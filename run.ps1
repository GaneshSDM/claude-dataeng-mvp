# Starts backend (:8000) and frontend (:8501). Run from repo root.
$root = $PSScriptRoot
$py = "C:\Users\Localuser\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "Set-Location '$root'; & '$py' -m uvicorn backend.main:app --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "Set-Location '$root'; & '$py' -m streamlit run frontend/Home.py --server.port 8501"
Write-Host "Backend: http://127.0.0.1:8000  Frontend: http://127.0.0.1:8501"
