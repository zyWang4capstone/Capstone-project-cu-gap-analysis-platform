@echo off
setlocal EnableExtensions

rem ===========================================
rem 04_Reset_workspace.bat
rem - Sticky console
rem - 1) Soft: clear reports only
rem - 2) Full: clear reports + delete data folder
rem ===========================================

rem 0) Relaunch once into sticky console
if "%~1"=="--stay" goto after_stay
start "" "%ComSpec%" /k "%~f0" --stay %*
exit /b
:after_stay
shift

rem 1) Go to project root
cd /d "%~dp0"
set "ROOT=%CD%"
echo [INFO] Project root: %ROOT%

rem 2) Choose mode
echo.
echo Choose reset mode:
echo   1 = Soft reset   (clear reports only)
echo   2 = Full reset   (clear reports AND delete the entire data folder)
set /p MODE=Enter 1 or 2: 

if "%MODE%"=="1" (set "DO_DATA=0") else if "%MODE%"=="2" (set "DO_DATA=1") else (
  echo [ERROR] Invalid choice.
  pause
  exit /b 1
)

echo.
echo This action cannot be undone.
set /p CONFIRM=Type YES to confirm: 
if /I not "%CONFIRM%"=="YES" (
  echo [INFO] Aborted by user.
  pause
  exit /b 0
)

rem 3) Wipe outputs under reports\... and report\...
call :wipe_reports "reports"
call :wipe_reports "report"

rem 4) Handle data\ (full reset only)
if "%DO_DATA%"=="1" (
  echo.
  echo [WARN] FULL reset selected: deleting entire data folder...
  if exist "data" rmdir /s /q "data"
  mkdir "data" 2>nul
) else (
  echo.
  echo [INFO] Soft reset: data folder is kept.
)

echo.
echo [OK] Reset complete.
echo You can now copy the NEW ZIP into data\ and run ② Run gap analysis.bat
pause
exit /b 0

:wipe_reports
set "BASE=%~1"
if not exist "%BASE%" goto :eof
call :wipe_dir "%BASE%\task1\cleaned"
call :wipe_dir "%BASE%\task1\diff"
call :wipe_dir "%BASE%\task1\eda"
call :wipe_dir "%BASE%\task2"
call :del_glob "%BASE%\task1" "*.csv"
call :del_glob "%BASE%\task1" "*.parquet"
call :del_glob "%BASE%\task1" "*.zip"
call :del_glob "%BASE%\task2" "*.csv"
call :del_glob "%BASE%\task2" "*.parquet"
call :del_glob "%BASE%\task2" "*.zip"
goto :eof

:wipe_dir
set "D=%~1"
if exist "%D%" rmdir /s /q "%D%"
goto :eof

:del_glob
set "D=%~1"
set "P=%~2"
if exist "%D%\%P%" del /q /f "%D%\%P%"
goto :eof