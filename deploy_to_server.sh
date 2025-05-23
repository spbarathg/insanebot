#!/bin/bash

# Deployment script for Solana Trading Bot
echo "Deploying Solana Trading Bot to server..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "Error: Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create necessary directories
mkdir -p data logs config models

# Set proper permissions
chmod 755 src/
chmod 644 src/*.py
chmod 644 src/*/*.py 2>/dev/null || true
chmod 777 data/ logs/ config/

# Convert line endings for all Python files
if command -v dos2unix &> /dev/null; then
    echo "Converting line endings..."
    find . -name "*.py" -exec dos2unix {} \;
else
    echo "dos2unix not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y dos2unix
        find . -name "*.py" -exec dos2unix {} \;
    elif command -v yum &> /dev/null; then
        sudo yum install -y dos2unix
        find . -name "*.py" -exec dos2unix {} \;
    else
        echo "Please install dos2unix manually and run: find . -name '*.py' -exec dos2unix {} \;"
    fi
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user -r requirements.txt

# Test syntax of main files
echo "Testing Python syntax..."
python3 -m py_compile src/main.py
if [ $? -eq 0 ]; then
    echo "✓ src/main.py syntax OK"
else
    echo "✗ src/main.py has syntax errors"
    exit 1
fi

# Test the bot in dry run mode
echo "Testing bot startup..."
timeout 10 python3 src/main.py || true

echo "Deployment complete!"
echo ""
echo "To run the bot:"
echo "  python3 src/main.py"
echo ""
echo "To run in background:"
echo "  nohup python3 src/main.py > logs/bot_output.log 2>&1 &" 