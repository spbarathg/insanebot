#!/bin/bash

# Enhanced Ant Bot - Docker Deployment Script
# Run this on your Ubuntu server

echo "🎯 Enhanced Ant Bot - Docker Deployment"
echo "=================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ ERROR: .env file not found!"
    echo "Please create .env file with your API keys:"
    echo "QUICKNODE_ENDPOINT_URL=your-quicknode-endpoint"
    echo "HELIUS_API_KEY=your-helius-key"
    echo "JUPITER_API_KEY=your-jupiter-key"
    echo "PRIVATE_KEY=your-wallet-private-key"
    exit 1
fi

# Check if QUICKNODE_ENDPOINT_URL is set
if ! grep -q "QUICKNODE_ENDPOINT_URL=" .env; then
    echo "⚠️  WARNING: QUICKNODE_ENDPOINT_URL not found in .env"
    echo "Add: QUICKNODE_ENDPOINT_URL=your-quicknode-endpoint-here"
fi

echo "✅ Environment file found"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs models
chmod 777 data logs
chmod 755 models

# Build and start the Enhanced Ant Bot
echo "🔨 Building Enhanced Ant Bot..."
docker compose build enhanced-ant-bot

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

# Start all services
echo "🚀 Starting Enhanced Ant Bot with monitoring..."
docker compose up -d

# Check service status
echo "📊 Service Status:"
docker compose ps

echo ""
echo "🎉 Enhanced Ant Bot Deployment Complete!"
echo ""
echo "📝 Access Points:"
echo "   Enhanced Ant Bot: Running in container 'enhanced-ant-bot'"
echo "   Grafana Dashboard: http://your-server:3000 (admin/admin)"
echo "   Prometheus: http://your-server:9090"
echo "   System Metrics: http://your-server:9100"
echo ""
echo "📋 Useful Commands:"
echo "   View logs: docker compose logs -f enhanced-ant-bot"
echo "   Stop all: docker compose down"
echo "   Restart: docker compose restart enhanced-ant-bot"
echo "   Update: git pull && docker compose build && docker compose up -d"
echo "" 