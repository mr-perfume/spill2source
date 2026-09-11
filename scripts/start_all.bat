@echo off
REM ===========================================================================
REM  Starts every service in its own window. Run this from the project root:
REM      scripts\start_all.bat
REM  Close the four windows to stop everything.
REM ===========================================================================
setlocal
cd /d "%~dp0.."
set ROOT=%CD%

echo Project root: %ROOT%
echo.

REM MongoDB must already be running. If it is installed as a Windows service:
REM     net start MongoDB
sc query MongoDB | find "RUNNING" >nul 2>&1
if errorlevel 1 (
  echo [warn] The MongoDB service does not look like it is running.
  echo        Start it with:  net start MongoDB
  echo        Or run mongod manually, then re-run this script.
  echo.
)

echo Starting detection service on port 8001...
start "detection :8001" cmd /k "cd /d %ROOT% && python backend\detection_service\main.py"
timeout /t 3 /nobreak >nul

echo Starting backtrack service on port 8002...
start "backtrack :8002" cmd /k "cd /d %ROOT% && python backend\backtrack_service\main.py"
timeout /t 2 /nobreak >nul

echo Starting AIS service on port 8003...
start "ais :8003" cmd /k "cd /d %ROOT% && python backend\ais_service\main.py"
timeout /t 2 /nobreak >nul

echo Starting gateway on port 4000...
start "gateway :4000" cmd /k "cd /d %ROOT%\backend\gateway && node server.js"
timeout /t 2 /nobreak >nul

echo Starting frontend on port 5173...
start "frontend :5173" cmd /k "cd /d %ROOT%\frontend && npm run dev"

echo.
echo All five processes launched. Give the detection service about 15 seconds
echo to load the U-Net, then open:
echo.
echo     http://localhost:5173
echo.
echo Check everything is up with:  python scripts\check_health.py
endlocal
