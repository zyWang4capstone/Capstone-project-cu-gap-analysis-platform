@echo off
setlocal EnableExtensions

REM ============================================
REM Capstone one-click runner (absolute script path, no fragile pre-check)
REM - Relaunch once into persistent console (/k)
REM - cd /d to script dir; sanitize inputs
REM - Resolve absolute path to tools\run_all_test.py and run it
REM - Log to run_log.txt; on failure, tail log + dir tools
REM ============================================

REM ---- 0) Relaunch once into sticky console ----
if /I "%~1" NEQ "--stay" (
  start "" "%ComSpec%" /k "%~f0" --stay %*
  exit /b
) else (
  shift
)

REM ---- 1) Anchor to this .bat folder & init log ----
cd /d "%~dp0"
set "THIS_DIR=%CD%"
set "LOG=%THIS_DIR%\run_log.txt"
call :log "===== Capstone Runner ====="
call :log "START: %DATE% %TIME%"
call :log "THIS_DIR=%THIS_DIR%"

REM ---- 2) Collect args (after --stay consumed) ----
set "VAL_COL=%~1"
set "VAL_MIN=%~2"
set "VAL_MAX=%~3"

if not defined VAL_COL set /p VAL_COL=Enter value column name (e.g., Cu_ppm):
if not defined VAL_MIN set /p VAL_MIN=Enter min value (e.g., 0):
if not defined VAL_MAX set /p VAL_MAX=Enter max value (e.g., 10000):

REM sanitize: strip outer quotes if the user pasted them
set "VAL_COL=%VAL_COL:"=%"
set "VAL_MIN=%VAL_MIN:"=%"
set "VAL_MAX=%VAL_MAX:"=%"

echo [ARGS] --value-col "%VAL_COL%" --value-min %VAL_MIN% --value-max %VAL_MAX%
call :log "[ARGS] value_col='%VAL_COL%'  value_min=%VAL_MIN%  value_max=%VAL_MAX%"

REM ---- 3) Find env python (capstone-cu) across common prefixes ----
set "ENV_NAME=capstone-cu"
set "PY_EXE="
call :try_py "%LOCALAPPDATA%\Capstone\Miniforge3"
call :try_py "%LOCALAPPDATA%\miniforge3"
call :try_py "C:\Users\%USERNAME%\miniforge3"
call :try_py "C:\Miniforge3"
call :try_py "%ProgramData%\Miniforge3"

echo [INFO] PY_EXE=%PY_EXE%
call :log "[INFO] PY_EXE=%PY_EXE%"
if not defined PY_EXE (
  echo [ERROR] Could not locate %ENV_NAME%\python.exe
  call :log "[ERROR] Could not locate %ENV_NAME%\python.exe"
  goto :END_FAIL
)

REM ---- 4) Resolve absolute path of tools\run_all_test.py (no pre-check) ----
set "PY_SCRIPT_REL=tools\run_all_test.py"
set "PY_SCRIPT="
for %%F in ("%PY_SCRIPT_REL%") do set "PY_SCRIPT=%%~fF"
echo [INFO] PY_SCRIPT=%PY_SCRIPT%
call :log "[INFO] PY_SCRIPT=%PY_SCRIPT%"

REM ---- 5) Run pipeline (adjust flags if your CLI names differ) ----
echo [STEP] Running pipeline...
call :log "[STEP] Running: ""%PY_EXE%"" ""%PY_SCRIPT%"" --value-col ""%VAL_COL%"" --value-min %VAL_MIN% --value-max %VAL_MAX%"

"%PY_EXE%" "%PY_SCRIPT%" ^
  --value-col "%VAL_COL%" ^
  --value-min %VAL_MIN% ^
  --value-max %VAL_MAX%  1>>"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo [ERROR] Pipeline failed. Exit code=%RC%
  call :log "[ERROR] Pipeline failed. Exit code=%RC%"
  goto :END_FAIL
)

echo [OK] Completed. Outputs under reports\task1\cleaned\ (and related folders).
call :log "[OK] Completed."
goto :END_OK

:END_FAIL
echo.
echo [DIR] Listing of tools\ (for diagnosis):
dir /a "tools"
call :log "[DIR] tools listing below:"
dir /a "tools" >>"%LOG%"

echo.
echo Log saved to: %LOG%
where powershell >nul 2>nul
if %ERRORLEVEL%==0 (
  powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 120"
) else (
  type "%LOG%"
)
pause
exit /b 1

:END_OK
echo.
echo Log saved to: %LOG%"
pause
exit /b 0

REM ---- helper: log one line ----
:log
>>"%LOG%" echo %~1
goto :eof

REM ---- helper: try to set PY_EXE from a given prefix ----
:try_py
if defined PY_EXE goto :eof
set "PREFIX=%~1"
if exist "%PREFIX%\envs\%ENV_NAME%\python.exe" (
  set "PY_EXE=%PREFIX%\envs\%ENV_NAME%\python.exe"
  call :log "[INFO] Found env at: %PREFIX%"
)
goto :eof