#!/bin/bash

# Deploy script for DigitalOcean droplet
# Usage: ./deploy_digitalocean.sh <droplet_name> <region>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <droplet_name> <region>"
    exit 1
fi

DROPLET_NAME=$1
REGION=$2

# Create droplet
echo "Creating droplet..."
doctl compute droplet create $DROPLET_NAME \
    --size s-1vcpu-1gb \
    --image ubuntu-20-04-x64 \
    --region $REGION \
    --ssh-keys $(doctl compute ssh-key list --format ID --no-header)

# Wait for droplet to be ready
echo "Waiting for droplet to be ready..."
sleep 30

# Get droplet IP
IP=$(doctl compute droplet get $DROPLET_NAME --format PublicIPv4 --no-header)

# Copy files
echo "Copying files..."
scp -r src/* root@$IP:/root/bot/
scp requirements.txt root@$IP:/root/bot/
scp .env root@$IP:/root/bot/

# Setup environment
echo "Setting up environment..."
ssh root@$IP << 'ENDSSH'
    cd /root/bot
    apt-get update
    apt-get install -y python3-pip
    pip3 install -r requirements.txt
    pip3 install gunicorn
    systemctl enable bot
    systemctl start bot
ENDSSH

echo "Deployment complete! Bot is running at $IP" 