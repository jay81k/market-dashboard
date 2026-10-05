@echo off
setlocal

rem Always run from the folder this .bat file is actually in,
rem regardless of where it's double-clicked from.
cd /d "%~dp0"

echo Checking for updates from GitHub...
where git >nul 2>&1
if %errorlevel%==0 (
    git pull --ff-only
    if errorlevel 1 (
        echo.
        echo [!] Could not fast-forward - you may have local changes, or you're
        echo     offline. Continuing with the data already on disk.
        echo.
    )
) else (
    echo [!] git not found on PATH - skipping update check.
    echo     Continuing with the data already on disk.
)

echo.
echo Starting local server on port 8000...
start "Dashboard Server - close this window to stop" /min cmd /k "python -m http.server 8000"

rem Give the server a moment to actually bind the port before we
rem try to open a browser tab pointing at it.
timeout /t 2 /nobreak >nul

start "" "http://localhost:8000"

echo.
echo Dashboard should now be open at http://localhost:8000
echo The server is running in a separate minimized window titled
echo "Dashboard Server" - close that window when you're done to stop it.
echo.
pause
