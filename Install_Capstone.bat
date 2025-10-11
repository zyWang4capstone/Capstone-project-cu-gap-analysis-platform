@echo off
setlocal enabledelayedexpansion

REM ====== Config ======
set "THIS_DIR=%~dp0"
set "CAP_DIR=%LOCALAPPDATA%\Capstone"
set "MF_HOME=%CAP_DIR%\Miniforge3"
set "ENV_NAME=capstone-cu"
set "ENV_DIR=%MF_HOME%\envs\%ENV_NAME%"
set "MF_EXE=%THIS_DIR%Miniforge3-Windows-x86_64.exe"
set "PACK_ZIP=%THIS_DIR%capstone-cu-packed.zip"
set "ENV_YML=%THIS_DIR%environment.yml"
set "REQ_TXT=%THIS_DIR%requirements.txt"
set "LOG=%THIS_DIR%install_log.txt"

echo ===== Capstone One-Click Installer ===== > "%LOG%"
echo THIS_DIR=%THIS_DIR%>>"%LOG%"
echo MF_HOME=%MF_HOME%>>"%LOG%"
echo ENV_DIR=%ENV_DIR%>>"%LOG%"
echo ========================================>>"%LOG%"

REM 1) Install Miniforge silently if missing
if not exist "%MF_HOME%\condabin\conda.bat" (
  echo [INFO] Installing Miniforge...>>"%LOG%"
  if exist "%MF_EXE%" (
    "%MF_EXE%" /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=%MF_HOME%
  ) else (
    echo [INFO] No local installer. Downloading from repo...>>"%LOG%"
    powershell -Command "Invoke-WebRequest -UseBasicParsing https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe -OutFile '%MF_EXE%'" || (
      echo [ERR ] Download Miniforge failed. See %LOG%
      exit /b 1
    )
    "%MF_EXE%" /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=%MF_HOME%
  )
)

REM 2) If packed env exists (offline), unpack first
if not exist "%ENV_DIR%" (
  if exist "%PACK_ZIP%" (
    echo [INFO] Unpacking prebuilt env...>>"%LOG%"
    powershell -Command "Expand-Archive -Force -Path '%PACK_ZIP%' -DestinationPath '%ENV_DIR%'" || (
      echo [ERR ] Expand-Archive failed.>>"%LOG%"
      exit /b 1
    )
    call "%MF_HOME%\condabin\conda.bat" run -p "%ENV_DIR%" python -c "import conda_pack; import os" 2>nul
    REM if conda-unpack exists, run it (fix prefixes)
    if exist "%ENV_DIR%\Scripts\conda-unpack.exe" (
      call "%ENV_DIR%\Scripts\conda-unpack.exe"
    )
  ) else (
    REM 3) Create env from YAML (preferred) or requirements.txt
    if exist "%ENV_YML%" (
      echo [INFO] Creating env from environment.yml...>>"%LOG%"
      call "%MF_HOME%\condabin\conda.bat" env create -y -n "%ENV_NAME%" -f "%ENV_YML%" || (
        echo [WARN] conda env create failed, trying mamba...>>"%LOG%"
        call "%MF_HOME%\condabin\conda.bat" install -y -n base -c conda-forge mamba && ^
        call "%MF_HOME%\condabin\mamba.bat" env create -y -n "%ENV_NAME%" -f "%ENV_YML%"
      )
    ) else if exist "%REQ_TXT%" (
      echo [INFO] Creating env from requirements.txt...>>"%LOG%"
      call "%MF_HOME%\condabin\conda.bat" create -y -n "%ENV_NAME%" -c conda-forge python=3.10 || exit /b 1
      call "%MF_HOME%\condabin\conda.bat" run -n "%ENV_NAME%" pip install -r "%REQ_TXT%" || exit /b 1
    ) else (
      echo [ERR ] No environment.yml or requirements.txt found.>>"%LOG%"
      exit /b 1
    )
  )
)

echo [OK  ] Environment ready at: %ENV_DIR%>>"%LOG%"
echo Done.
exit /b 0
