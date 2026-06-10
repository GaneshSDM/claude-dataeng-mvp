# DataEng Copilot — Multi-Agent Platform launcher
# Starts backend (:8000) and frontend (:8501) in separate PowerShell windows
$root = $PSScriptRoot
$py = "C:\Users\Localuser\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "⚡ DataEng Copilot — Multi-Agent Platform" -ForegroundColor Cyan
Write-Host ""

# Check deps
try { & $py -c "import fastapi, uvicorn, streamlit, duckdb, plotly" 2>$null; Write-Host "✅ Dependencies OK" -ForegroundColor Green }
catch { Write-Host "📦 Installing deps..."; pip install -r requirements.txt }

Write-Host "🚀 Starting backend (port 8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "Set-Location '$root'; & '$py' -m uvicorn backend.main:app --port 8000"

Start-Sleep 3

Write-Host "🚀 Starting frontend (port 8501)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "Set-Location '$root'; & '$py' -m streamlit run frontend/Home.py --server.port 8501"

Write-Host ""
Write-Host "✅ DataEng Copilot running!" -ForegroundColor Green
Write-Host "   Frontend : http://127.0.0.1:8501" -ForegroundColor White
Write-Host "   Backend  : http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "   Health   : http://127.0.0.1:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "Pages:" -ForegroundColor Cyan
Write-Host "   🏠  Home        — Landing + metrics + deck download"
Write-Host "   💬  Chat        — Natural language data questions → agents"
Write-Host "   🔧  Pipeline    — Visual pipeline builder (compose agent DAGs)"
Write-Host "   📊  Monitoring  — Agent activity dashboard (auto-refresh)"
Write-Host "   📈  ROI Calc    — ROI modeling with sliders"
Write-Host "   ✉️  Contact     — CTA"
Write-Host ""
Write-Host "🛑 Close the PowerShell windows to stop" -ForegroundColor Yellow