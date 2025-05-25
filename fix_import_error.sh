#!/bin/bash

# Fix Enhanced Ant Bot Import Error
# Run this on your Ubuntu server

echo "🔧 Fixing Enhanced Ant Bot Import Error"
echo "======================================"

# Stop the current container
echo "🛑 Stopping Enhanced Ant Bot..."
docker-compose down enhanced-ant-bot 2>/dev/null || true

# Fix the import error in enhanced_main_entry.py
echo "🔨 Fixing import error..."
sed -i 's/EnhancedAICoordinator/AICoordinator/g' enhanced_main_entry.py

# Verify the fix
echo "✅ Checking fix..."
if grep -q "AICoordinator" enhanced_main_entry.py && ! grep -q "EnhancedAICoordinator" enhanced_main_entry.py; then
    echo "✅ Import error fixed successfully!"
else
    echo "❌ Fix verification failed"
    exit 1
fi

# Rebuild and start
echo "🔨 Rebuilding Enhanced Ant Bot..."
docker-compose build --no-cache enhanced-ant-bot

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    echo "🚀 Starting Enhanced Ant Bot..."
    docker-compose up -d enhanced-ant-bot
    
    # Wait a moment for startup
    sleep 5
    
    # Check status
    echo "📊 Container Status:"
    docker ps --filter "name=enhanced-ant-bot"
    
    # Show logs
    echo "📋 Enhanced Ant Bot Logs:"
    docker-compose logs --tail=20 enhanced-ant-bot
    
    # Check if container is still running (not exiting)
    if docker ps --filter "name=enhanced-ant-bot" --filter "status=running" | grep -q enhanced-ant-bot; then
        echo ""
        echo "🎉 SUCCESS! Enhanced Ant Bot is now running properly!"
        echo ""
        echo "📋 Next steps:"
        echo "   View live logs: docker-compose logs -f enhanced-ant-bot"
        echo "   Check status: docker ps"
        echo "   Stop: docker-compose down"
    else
        echo ""
        echo "❌ Container exited. Check logs above for any remaining errors."
    fi
    
else
    echo "❌ Build failed!"
    exit 1
fi 