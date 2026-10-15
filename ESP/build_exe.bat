@echo off
REM Double-click this once to build ESP_LLM.exe (needs Python installed).
cd /d "%~dp0"
set PY=py
%PY% --version >nul 2>&1 || set PY=python
%PY% --version >nul 2>&1 || (
  echo Python was not found. Install it from python.org and tick "Add Python to PATH", then run this again.
  pause
  exit /b 1
)

echo Installing pyserial and pyinstaller (needs internet)...
%PY% -m pip install --retries 10 --timeout 60 --no-cache-dir pyserial pyinstaller
if errorlevel 1 (
  echo.
  echo The install failed - usually a dropped internet connection. Run this file again.
  echo If it keeps failing, try turning off any VPN, or use a different network.
  pause
  exit /b 1
)

echo Building ESP_LLM.exe ...
%PY% -m PyInstaller --onefile --console --name ESP_LLM --hidden-import serial.tools.list_ports_windows ESP_LLM.py
if errorlevel 1 (
  echo.
  echo The build failed - see the messages above.
  pause
  exit /b 1
)

if not exist dist\ESP_LLM.exe (
  echo Build finished but dist\ESP_LLM.exe was not found.
  pause
  exit /b 1
)
copy /Y dist\ESP_LLM.exe ESP_LLM.exe >nul
echo.
echo Done - ESP_LLM.exe is in this folder.
pause