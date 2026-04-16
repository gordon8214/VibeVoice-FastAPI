@echo off
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run: python install.py
    exit /b 1
)

:: Activate venv and launch via Python (batch can't reliably parse .env with JSON values)
call venv\Scripts\activate.bat
python start_server.py
