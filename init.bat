@echo off
setlocal enabledelayedexpansion

set LOGFILE=init_log.txt

REM 引数からPythonバージョンを取得
if "%~1"=="" (
    echo Error: No Python version specified.
    echo [ERROR] No Python version specified. >> %LOGFILE%
    exit /b 1
) else (
    set PYTHON_VERSION=%~1
)

echo Setup started at %date% %time% > %LOGFILE%
echo Setup started at %date% %time%

REM pyenv-win がインストールされているか確認
where pyenv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: pyenv-win is not installed. Please install pyenv-win manually.
    echo [ERROR] pyenv-win is not installed. >> %LOGFILE%
    exit /b 1
)

REM 指定された Python バージョンがインストールされているか確認
pyenv versions | findstr %PYTHON_VERSION% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing Python %PYTHON_VERSION% using pyenv...
    echo Installing Python %PYTHON_VERSION% using pyenv... >> %LOGFILE%
    pyenv install %PYTHON_VERSION%
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to install Python %PYTHON_VERSION%.
        echo [ERROR] Failed to install Python %PYTHON_VERSION%. >> %LOGFILE%
        exit /b 1
    )
    echo [SUCCESS] Python %PYTHON_VERSION% installed successfully. >> %LOGFILE%
)

REM 指定されたバージョンのPythonを取得
set PYTHON_PATH=%USERPROFILE%\.pyenv\pyenv-win\versions\%PYTHON_VERSION%\python.exe
if not exist "%PYTHON_PATH%" (
    echo Error: Specified Python version %PYTHON_VERSION% not found.
    echo [ERROR] Specified Python version %PYTHON_VERSION% not found. >> %LOGFILE%
    exit /b 1
)

REM 仮想環境を作成（すでに存在する場合は削除して再作成）
if exist venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)
echo Creating virtual environment using Python %PYTHON_VERSION%...
echo Creating virtual environment using Python %PYTHON_VERSION%... >> %LOGFILE%
"%PYTHON_PATH%" -m venv venv
if %ERRORLEVEL% neq 0 (
    echo Error: Virtual environment creation failed.
    echo [ERROR] Virtual environment creation failed. >> %LOGFILE%
    exit /b 1
) else (
    echo [SUCCESS] Virtual environment created successfully. >> %LOGFILE%
    echo Virtual environment created successfully.
)

REM 仮想環境をアクティブにする
echo Activating virtual environment...
echo Activating virtual environment... >> %LOGFILE%
call venv\Scripts\activate

REM 仮想環境がアクティブになったか確認
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Virtual environment activation failed.
    echo [ERROR] Virtual environment activation failed. >> %LOGFILE%
    exit /b 1
) else (
    echo [SUCCESS] Virtual environment activated. >> %LOGFILE%
    echo Virtual environment activated.
)

REM pipのアップグレード
echo Upgrading pip...
echo Upgrading pip... >> %LOGFILE%
python -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to upgrade pip.
    echo [ERROR] Failed to upgrade pip. >> %LOGFILE%
    exit /b 1
) else (
    echo [SUCCESS] pip upgraded successfully. >> %LOGFILE%
    echo pip upgraded successfully.
)

REM requirements.txt がある場合、パッケージをインストール
if exist requirements.txt (
    for /f %%i in ('findstr /R /N "^" requirements.txt ^| find /C ":"') do set REQUIREMENTS_LINES=%%i
    if !REQUIREMENTS_LINES! GTR 0 (
        echo Installing required libraries from requirements.txt...
        echo Installing required libraries from requirements.txt... >> %LOGFILE%
        pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
        if %ERRORLEVEL% neq 0 (
            echo Error: Failed to install required libraries from requirements.txt.
            echo [ERROR] Failed to install required libraries from requirements.txt. >> %LOGFILE%
            exit /b 1
        ) else (
            echo [SUCCESS] Required libraries installed successfully from requirements.txt. >> %LOGFILE%
            echo Required libraries installed successfully.
        )
    ) else (
        echo Skipping package installation. requirements.txt is empty.
        echo [INFO] Skipping package installation. requirements.txt is empty. >> %LOGFILE%
    )
) else (
    echo No requirements.txt found. Skipping package installation.
    echo [INFO] No requirements.txt found. >> %LOGFILE%
)

echo Setup completed at %date% %time%. >> %LOGFILE%
echo Setup completed at %date% %time%.

endlocal
