@echo off
setlocal EnableExtensions

rem ===========================================
rem One-click Streamlit launcher (ASCII-safe)
rem - Relaunch into sticky console (/k)
rem - cd to script dir
rem - Find env python
rem - Check/install streamlit via pip show
rem - Run start.py on chosen port (default 8501)
rem ===========================================

rem 0) Relaunch once into sticky console
if "%~1"=="--stay" goto after_stay
start "" "%ComSpec%" /k "%~f0" --stay %*
exit /b
:after_stay
shift

rem 1) Anchor to this folder
cd /d "%~dp0"
set "THIS_DIR=%CD%"
set "ENV_NAME=capstone-cu"

rem 2) Optional port arg
set "PORT=%~1"
if not defined PORT set "PORT=8501"

rem 3) Find env python across common prefixes
set "PY_EXE="
call :try_py "%LOCALAPPDATA%\Capstone\Miniforge3"
call :try_py "%LOCALAPPDATA%\miniforge3"
call :try_py "C:\Users\%USERNAME%\miniforge3"
call :try_py "C:\Miniforge3"
call :try_py "%ProgramData%\Miniforge3"

if not defined PY_EXE (
  echo [ERROR] Could not locate %ENV_NAME%\python.exe
  pause
  exit /b 1
)

rem 4) Check start.py presence
if not exist "%THIS_DIR%\start.py" (
  echo [ERROR] Missing start.py at %THIS_DIR%
  dir /a "%THIS_DIR%"
  pause
  exit /b 1
)

rem 5) Ensure streamlit is present (pip show)
echo [CHECK] streamlit in %ENV_NAME% ...
"%PY_EXE%" -m pip show streamlit >nul 2>&1
if "%ERRORLEVEL%"=="0" goto have_streamlit

echo [INFO] Installing streamlit into the environment
"%PY_EXE%" -m pip install --upgrade pip
if not "%ERRORLEVEL%"=="0" echo [ERROR] pip upgrade failed.& pause& exit /b 1
"%PY_EXE%" -m pip install streamlit
if not "%ERRORLEVEL%"=="0" echo [ERROR] streamlit install failed.& pause& exit /b 1

:have_streamlit
echo [RUN] "%PY_EXE%" -m streamlit run "%THIS_DIR%\start.py" --server.port %PORT%
echo [INFO] If the browser does not open, visit: http://localhost:%PORT%/
"%PY_EXE%" -m streamlit run "%THIS_DIR%\start.py" --server.port %PORT%

echo.
echo [EXIT] Streamlit process finished (or stopped).
pause
exit /b 0

:try_py
if defined PY_EXE goto :eof
set "PREFIX=%~1"
if exist "%PREFIX%\envs\%ENV_NAME%\python.exe" (
  set "PY_EXE=%PREFIX%\envs\%ENV_NAME%\python.exe"
  echo [INFO] Using env: %PREFIX%\envs\%ENV_NAME%
)
goto :eof