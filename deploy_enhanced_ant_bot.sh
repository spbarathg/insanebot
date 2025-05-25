#!/bin/bash

# Enhanced Ant Bot - Docker Deployment Script
# Run this on your Ubuntu server

echo "🎯 Enhanced Ant Bot - Docker Deployment"
echo "=================================="

# Function to cleanup if needed
cleanup_existing() {
    echo "🧹 Cleaning up existing containers..."
    docker compose down 2>/dev/null || true
    docker container stop enhanced-ant-bot 2>/dev/null || true
    docker container rm enhanced-ant-bot 2>/dev/null || true
    docker container stop trading-bot 2>/dev/null || true
    docker container rm trading-bot 2>/dev/null || true
}

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ ERROR: .env file not found!"
    echo ""
    if [ -f env.template ]; then
        echo "📋 Found env.template file. Copy it to .env and fill in your values:"
        echo "cp env.template .env"
        echo "nano .env"
    else
        echo "Please create .env file with your API keys:"
        echo "QUICKNODE_ENDPOINT_URL=your-quicknode-endpoint"
        echo "HELIUS_API_KEY=your-helius-key"
        echo "JUPITER_API_KEY=your-jupiter-key"
        echo "PRIVATE_KEY=your-wallet-private-key"
    fi
    exit 1
fi

# Check required environment variables
echo "🔍 Checking environment configuration..."
missing_vars=()

if ! grep -q "QUICKNODE_ENDPOINT_URL=" .env || grep -q "QUICKNODE_ENDPOINT_URL=your-" .env; then
    missing_vars+=("QUICKNODE_ENDPOINT_URL")
fi

if ! grep -q "HELIUS_API_KEY=" .env || grep -q "HELIUS_API_KEY=your-" .env; then
    missing_vars+=("HELIUS_API_KEY")
fi

if ! grep -q "PRIVATE_KEY=" .env || grep -q "PRIVATE_KEY=your-" .env; then
    missing_vars+=("PRIVATE_KEY")
fi

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "❌ ERROR: Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Please update your .env file with the actual values."
    exit 1
fi

echo "✅ Environment file validated"

# Cleanup any existing containers
cleanup_existing

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs models
chmod 777 data logs
chmod 755 models

# Check for port conflicts
echo "🔍 Checking for port conflicts..."
if lsof -i :8001 >/dev/null 2>&1; then
    echo "⚠️  Port 8001 is in use, but Enhanced Ant Bot doesn't need external ports"
fi

# Build the Enhanced Ant Bot
echo "🔨 Building Enhanced Ant Bot..."
docker compose build enhanced-ant-bot

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    echo "Try running: docker compose build --no-cache enhanced-ant-bot"
    exit 1
fi

# Start all services
echo "🚀 Starting Enhanced Ant Bot with monitoring..."
docker compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Services started successfully!"
else
    echo "❌ Failed to start services!"
    echo "Check logs with: docker compose logs enhanced-ant-bot"
    exit 1
fi

# Wait a moment for services to initialize
sleep 5

# Check service status
echo "📊 Service Status:"
docker compose ps

# Check Enhanced Ant Bot logs
echo "📋 Enhanced Ant Bot Status:"
docker compose logs --tail=10 enhanced-ant-bot

echo ""
echo "🎉 Enhanced Ant Bot Deployment Complete!"
echo ""
echo "📝 Access Points:"
echo "   Enhanced Ant Bot: Running in container 'enhanced-ant-bot'"
echo "   Grafana Dashboard: http://$(hostname -I | awk '{print $1}'):3000 (admin/admin)"
echo "   Prometheus: http://$(hostname -I | awk '{print $1}'):9090"
echo "   System Metrics: http://$(hostname -I | awk '{print $1}'):9100"
echo ""
echo "📋 Useful Commands:"
echo "   View logs: docker compose logs -f enhanced-ant-bot"
echo "   Stop all: docker compose down"
echo "   Restart: docker compose restart enhanced-ant-bot"
echo "   Update: git pull && docker compose build && docker compose up -d"
echo "   Shell access: docker compose exec enhanced-ant-bot bash"
echo "" 