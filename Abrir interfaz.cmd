@echo off
setlocal
set "PROSODIA_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PROSODIA_PYTHON%" (
    echo No se encontro Python 3.12 en "%PROSODIA_PYTHON%".
    pause
    exit /b 1
)
"%PROSODIA_PYTHON%" "%~dp0interfaz.py"
if errorlevel 1 pause
