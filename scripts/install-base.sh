#!/bin/bash
# Install base development tools on Ubuntu
# Run with: sudo bash install-base.sh

set -e

echo "=== Updating system ==="
apt update && apt upgrade -y

echo "=== Installing base tools ==="
apt install -y git curl wget build-essential htop tmux tree jq unzip

echo "=== Installing uv (Python package manager) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "=== Installing VS Code ==="
snap install code --classic

echo "=== Installing GitHub CLI ==="
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
apt update && apt install -y gh

echo "=== Done! ==="
echo "Run: source ~/.bashrc"