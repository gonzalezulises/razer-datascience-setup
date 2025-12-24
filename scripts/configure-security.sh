#!/bin/bash
# Configure security hardening
# Run with: sudo bash configure-security.sh

set -e

echo "=== Configuring UFW Firewall ==="
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 41641/udp comment "Tailscale"
ufw --force enable
ufw status verbose

echo "=== Installing Fail2ban ==="
apt install -y fail2ban
cat > /etc/fail2ban/jail.local << "EOF"
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5
banaction = ufw

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF
systemctl enable fail2ban
systemctl restart fail2ban

echo "=== Configuring SSH Hardening ==="
cat > /etc/ssh/sshd_config.d/hardening.conf << "EOF"
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthenticationMethods publickey
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
EOF
systemctl restart sshd

echo "=== Installing ClamAV ==="
apt install -y clamav clamav-daemon
systemctl stop clamav-freshclam
freshclam
systemctl start clamav-freshclam

echo "=== Security configuration complete! ==="