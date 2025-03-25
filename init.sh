#!/bin/bash

LOGFILE="init_log.txt"

# 引数からPythonバージョンを取得
if [ -z "$1" ]; then
    echo "Error: No Python version specified."
    echo "[ERROR] No Python version specified." >> "$LOGFILE"
    exit 1
else
    PYTHON_VERSION=$1
fi

echo "Setup started at $(date)" > "$LOGFILE"
echo "Setup started at $(date)"

# CPU アーキテクチャ確認（あくまでログ目的）
ARCH=$(uname -m)
echo "Running in native $ARCH mode"

# Homebrew のインストール確認
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    echo "Installing Homebrew..." >> "$LOGFILE"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Xcode Command Line Tools のインストール
if ! xcode-select -p &>/dev/null; then
    echo "Installing Xcode Command Line Tools..."
    echo "Installing Xcode Command Line Tools..." >> "$LOGFILE"
    xcode-select --install
fi

# Homebrew パッケージのインストール
echo "Updating Homebrew and installing dependencies..."
brew update && brew upgrade
brew install pyenv openssl readline sqlite3 xz zlib tcl-tk git curl

# `pyenv` の環境変数を `.bashrc` または `.zshrc` に追加
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q 'export PYENV_ROOT="$HOME/.pyenv"' "$SHELL_RC"; then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_RC"
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_RC"
    echo 'eval "$(pyenv init --path)"' >> "$SHELL_RC"
    echo 'eval "$(pyenv init -)"' >> "$SHELL_RC"
    echo "Added pyenv settings to $SHELL_RC"
fi

# 設定を即時反映
source "$SHELL_RC"

# Apple Silicon 対応のビルド設定（zlibやsqliteのパス明示）
export LDFLAGS="-L/opt/homebrew/opt/zlib/lib -L/opt/homebrew/opt/sqlite/lib"
export CPPFLAGS="-I/opt/homebrew/opt/zlib/include -I/opt/homebrew/opt/sqlite/include"
export PKG_CONFIG_PATH="/opt/homebrew/opt/zlib/lib/pkgconfig:/opt/homebrew/opt/sqlite/lib/pkgconfig"

# Python のインストール
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION using pyenv..."
    echo "Installing Python $PYTHON_VERSION using pyenv..." >> "$LOGFILE"

    env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install "$PYTHON_VERSION"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to install Python $PYTHON_VERSION."
        echo "[ERROR] Failed to install Python $PYTHON_VERSION." >> "$LOGFILE"
        exit 1
    fi
    echo "[SUCCESS] Python $PYTHON_VERSION installed successfully." >> "$LOGFILE"
fi

# グローバルに指定
pyenv global "$PYTHON_VERSION"

PYTHON_PATH="$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python"

if [ ! -x "$PYTHON_PATH" ]; then
    echo "Error: Specified Python version $PYTHON_VERSION not found."
    echo "[ERROR] Specified Python version $PYTHON_VERSION not found." >> "$LOGFILE"
    exit 1
fi

# 仮想環境の作成
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

echo "Creating virtual environment using Python $PYTHON_VERSION..."
"$PYTHON_PATH" -m venv venv
if [ $? -ne 0 ]; then
    echo "Error: Virtual environment creation failed."
    echo "[ERROR] Virtual environment creation failed." >> "$LOGFILE"
    exit 1
fi

# 仮想環境をアクティベート
. venv/bin/activate

# requirements.txt が存在すればインストール
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    echo "Installing packages from requirements.txt..." >> "$LOGFILE"
    
    pip install --upgrade pip
    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo "Error: Failed to install packages from requirements.txt."
        echo "[ERROR] Failed to install packages from requirements.txt." >> "$LOGFILE"
        exit 1
    fi

    echo "[SUCCESS] Installed packages from requirements.txt." >> "$LOGFILE"
else
    echo "No requirements.txt found. Skipping package installation."
    echo "[INFO] No requirements.txt found. Skipped package installation." >> "$LOGFILE"
fi

echo "Setup completed at $(date)." >> "$LOGFILE"
echo "Setup completed at $(date)."