#!/bin/bash
# Simple installation script

echo "Installing sensing area detection package..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .

echo "Installation completed!"
echo "To activate the environment, run: source venv/bin/activate"