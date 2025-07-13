#!/bin/bash

echo "Installing sensing area detection package..."

# Create virtual environment with Python 3.11 if it doesn't exist
if [ ! -d "sensing-env" ]; then
    python3.11 -m venv sensing-env
fi

# Activate virtual environment
source sensing-env/bin/activate

# Upgrade pip to avoid legacy issues
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .

echo "Installation completed!"
echo "To activate the environment, run: source sensing-env/bin/activate"
