#!/bin/bash

# Exit on error
set -o errexit

# Install ffmpeg
echo "Installing ffmpeg"
apt-get update && apt-get install -y ffmpeg

# Install your Python dependencies
echo "Installing Python dependencies"
pip install -r requirements.txt

echo "Build script finished"
