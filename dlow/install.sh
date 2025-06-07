#!/bin/bash

# Function to check if conda is installed
check_conda_installed() {
    if command -v conda &> /dev/null; then
        echo "✅ Conda is already installed."
        return 0
    else
        echo "❌ Conda not found. Installing Miniconda..."
        return 1
    fi
}

# Function to install Miniconda
install_miniconda() {
    # Choose installer based on OS
    OS=$(uname)
    if [[ "$OS" == "Linux" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OS" == "Darwin" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    else
        echo "❌ Unsupported OS: $OS"
        exit 1
    fi

    INSTALLER="Miniconda3-latest.sh"

    # Download installer
    curl -L "$MINICONDA_URL" -o "$INSTALLER"
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to download Miniconda installer."
        exit 1
    fi

    # Run the installer silently
    bash "$INSTALLER" -b -p "$HOME/miniconda"
    if [[ $? -ne 0 ]]; then
        echo "❌ Miniconda installation failed."
        exit 1
    fi

    # Initialize conda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    "$HOME/miniconda/bin/conda" init

    echo "✅ Miniconda installed successfully. Please restart your terminal or run:"
    echo "   source ~/.bashrc  # or ~/.zshrc if using zsh"
}

# Main logic
check_conda_installed || install_miniconda

conda env create -f environment.yml

