#!/bin/bash

# Quick Fix for Enhanced Ant Bot Deployment Issues
# Run this immediately on your Ubuntu server

echo "🔧 Quick Fix - Enhanced Ant Bot Deployment"
echo "==========================================="

# Stop and remove ALL containers to resolve conflicts
echo "🛑 Stopping all containers..."
docker compose down 2>/dev/null || true
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true

# Remove any conflicting containers by name
echo "🧹 Cleaning up specific containers..."
docker container stop enhanced-ant-bot trading-bot 2>/dev/null || true
docker container rm enhanced-ant-bot trading-bot 2>/dev/null || true

# Check current .env file
echo "🔍 Checking .env configuration..."
if [ ! -f .env ]; then
    echo "❌ .env file missing!"
    echo "Creating basic .env template..."
    cat > .env << 'EOF'
# Add your actual API keys here
QUICKNODE_ENDPOINT_URL=your-quicknode-endpoint-here
HELIUS_API_KEY=your-helius-key-here
JUPITER_API_KEY=
PRIVATE_KEY=your-private-key-here
WALLET_ADDRESS=
GROK_API_KEY=
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin123
EOF
    echo "📝 Created .env template - PLEASE UPDATE WITH YOUR ACTUAL KEYS!"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs models
chmod 777 data logs
chmod 755 models

# Show current git status
echo "📋 Git Status:"
git status --porcelain

# Pull latest changes if working directory is clean
if [ -z "$(git status --porcelain)" ]; then
    echo "🔄 Pulling latest changes..."
    git pull
else
    echo "⚠️  Working directory has changes - skipping git pull"
fi

# Build only the enhanced-ant-bot service
echo "🔨 Building Enhanced Ant Bot (clean build)..."
docker compose build --no-cache enhanced-ant-bot

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Start only enhanced-ant-bot first
    echo "🚀 Starting Enhanced Ant Bot..."
    docker compose up -d enhanced-ant-bot
    
    if [ $? -eq 0 ]; then
        echo "✅ Enhanced Ant Bot started!"
        
        # Wait a moment
        sleep 3
        
        # Check status
        echo "📊 Container Status:"
        docker ps --filter "name=enhanced-ant-bot"
        
        # Show logs
        echo "📋 Enhanced Ant Bot Logs (last 20 lines):"
        docker compose logs --tail=20 enhanced-ant-bot
        
        echo ""
        echo "🎉 Quick Fix Complete!"
        echo "Enhanced Ant Bot should now be running without port conflicts."
        echo ""
        echo "Next steps:"
        echo "1. Update .env with your actual API keys"
        echo "2. Restart: docker compose restart enhanced-ant-bot"
        echo "3. Start monitoring: docker compose up -d"
        
    else
        echo "❌ Failed to start Enhanced Ant Bot"
        echo "Check logs: docker compose logs enhanced-ant-bot"
    fi
else
    echo "❌ Build failed!"
    echo "Check for issues and try: docker system prune -f"
fi

echo ""
echo "📋 Useful commands:"
echo "   View logs: docker compose logs -f enhanced-ant-bot"
echo "   Restart: docker compose restart enhanced-ant-bot"
echo "   Shell access: docker compose exec enhanced-ant-bot bash"
echo "   Stop: docker compose down" 