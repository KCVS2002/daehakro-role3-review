@echo off
setlocal
if defined DAEHAKRO_PYTHON goto python_configured
set "DAEHAKRO_PYTHON=%~dp0..\..\..\..\.venv-rl-cuda\Scripts\python.exe"
if exist "%DAEHAKRO_PYTHON%" goto python_configured
set "DAEHAKRO_PYTHON=python"
:python_configured
"%DAEHAKRO_PYTHON%" -B "%~dp0replay_policy.py" --gui
if errorlevel 1 pause
