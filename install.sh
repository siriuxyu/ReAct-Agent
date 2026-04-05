#!/usr/bin/env bash
set -e

CONDA_BASE=/home/siriux/anaconda3
CONDA_ENV=agent
PROJECT_DIR=/home/siriux/Projects/Cliriux

# Activate conda env
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

echo ">>> Installing Python dependencies..."
pip install "python-telegram-bot[webhooks]" httpx

echo ">>> Installing systemd services..."
sudo cp "$PROJECT_DIR/cliriux-agent.service" /etc/systemd/system/
sudo cp "$PROJECT_DIR/cliriux-telegram.service" /etc/systemd/system/

echo ">>> Enabling and starting services..."
sudo systemctl daemon-reload
sudo systemctl enable cliriux-agent cliriux-telegram
sudo systemctl start cliriux-agent

echo ">>> Waiting for agent server to start..."
sleep 5

sudo systemctl start cliriux-telegram

echo ""
sudo systemctl status cliriux-agent cliriux-telegram --no-pager
echo ""
echo "Done. Send /start to your Telegram bot to test."
