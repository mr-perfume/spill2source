@echo off
REM Frees the five ports this project uses, in case a window was closed
REM without the process exiting.
for %%P in (4000 5173 8001 8002 8003) do (
  for /f "tokens=5" %%A in ('netstat -ano ^| findstr ":%%P" ^| findstr "LISTENING"') do (
    echo Stopping PID %%A on port %%P
    taskkill /F /PID %%A >nul 2>&1
  )
)
echo Ports 4000, 5173, 8001, 8002, 8003 released.
