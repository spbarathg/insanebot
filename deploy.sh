#!/bin/bash
# Enhanced Solana Trading Bot - Production Deployment Script
# Version: 2.0
# Usage: ./deploy.sh [environment] [mode]
# Example: ./deploy.sh production live

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/deployment.log"
ENVIRONMENT="${1:-development}"
MODE="${2:-simulation}"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}" | tee -a "$LOG_FILE"
}

print_banner() {
    echo -e "${GREEN}"
    echo "=============================================================="
    echo "🤖 ENHANCED SOLANA TRADING BOT - PRODUCTION DEPLOYMENT"
    echo "=============================================================="
    echo "Environment: $ENVIRONMENT"
    echo "Mode: $MODE"
    echo "Timestamp: $(date)"
    echo "=============================================================="
    echo -e "${NC}"
}

check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if .env file exists
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        warning ".env file not found. Creating from template..."
        cp "$SCRIPT_DIR/env.template" "$SCRIPT_DIR/.env"
        error "Please configure the .env file with your API keys and settings before deployment."
    fi
    
    # Check critical environment variables
    source "$SCRIPT_DIR/.env"
    
    if [ -z "$HELIUS_API_KEY" ] || [ "$HELIUS_API_KEY" = "your-helius-api-key-here" ]; then
        error "HELIUS_API_KEY not configured in .env file"
    fi
    
    if [ -z "$QUICKNODE_ENDPOINT_URL" ] || [ "$QUICKNODE_ENDPOINT_URL" = "your-quicknode-endpoint-here" ]; then
        error "QUICKNODE_ENDPOINT_URL not configured in .env file"
    fi
    
    if [ "$MODE" = "live" ] && [ "$SIMULATION_MODE" = "true" ]; then
        error "Cannot deploy in live mode with SIMULATION_MODE=true. Set SIMULATION_MODE=false for live trading."
    fi
    
    log "✅ Prerequisites check passed"
}

setup_directories() {
    log "📁 Setting up directories..."
    
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/data"
    mkdir -p "$SCRIPT_DIR/monitoring"
    mkdir -p "$SCRIPT_DIR/monitoring/grafana/dashboards"
    mkdir -p "$SCRIPT_DIR/monitoring/grafana/provisioning"
    mkdir -p "$SCRIPT_DIR/backups"
    
    # Set permissions
    chmod 755 "$SCRIPT_DIR/logs"
    chmod 755 "$SCRIPT_DIR/data"
    
    log "✅ Directories setup complete"
}

create_monitoring_config() {
    log "📊 Creating monitoring configuration..."
    
    # Prometheus configuration
    cat > "$SCRIPT_DIR/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: []
EOF
    
    # Prometheus alert rules
    cat > "$SCRIPT_DIR/monitoring/alert_rules.yml" << EOF
groups:
  - name: trading_bot_alerts
    rules:
      - alert: TradingBotDown
        expr: up{job="trading-bot"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Trading bot is down"
          description: "The trading bot has been down for more than 2 minutes"
      
      - alert: HighErrorRate
        expr: rate(trading_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 10% for the last 5 minutes"
EOF
    
    log "✅ Monitoring configuration created"
}

create_systemd_service() {
    if [ "$ENVIRONMENT" = "production" ]; then
        log "🔧 Creating systemd service..."
        
        sudo tee /etc/systemd/system/solana-trading-bot.service > /dev/null << EOF
[Unit]
Description=Solana Trading Bot
After=docker.service
Requires=docker.service

[Service]
Type=forking
Restart=unless-stopped
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        sudo systemctl enable solana-trading-bot.service
        
        log "✅ Systemd service created and enabled"
    fi
}

deploy_application() {
    log "🚀 Deploying application..."
    
    # Pull latest images
    docker-compose pull
    
    # Build the application
    docker-compose build --no-cache
    
    # Start services
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose up -d
    else
        docker-compose up -d
    fi
    
    # Wait for services to be ready
    log "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log "✅ Application deployed successfully"
}

check_service_health() {
    log "🏥 Checking service health..."
    
    # Check if containers are running
    if ! docker-compose ps | grep -q "Up"; then
        error "Some containers are not running properly"
    fi
    
    # Check trading bot health (if health endpoint exists)
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            log "✅ Trading bot health check passed"
            break
        else
            warning "Health check attempt $attempt/$max_attempts failed, retrying..."
            sleep 10
            ((attempt++))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        warning "Health check failed after $max_attempts attempts"
    fi
}

show_status() {
    log "📊 System Status"
    echo ""
    echo "Container Status:"
    docker-compose ps
    echo ""
    echo "Resource Usage:"
    docker stats --no-stream
    echo ""
    echo "Recent Logs:"
    docker-compose logs --tail=20 trading-bot
    echo ""
    info "Access Points:"
    info "🤖 Trading Bot Logs: docker-compose logs -f trading-bot"
    info "📊 Prometheus: http://localhost:9090"
    info "📈 Grafana: http://localhost:3000 (admin/admin123)"
    info "💾 Redis: localhost:6379"
}

create_backup_script() {
    log "💾 Creating backup script..."
    
    cat > "$SCRIPT_DIR/backup.sh" << 'EOF'
#!/bin/bash
# Backup script for trading bot data

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r ./config "$BACKUP_DIR/"

# Backup data
cp -r ./data "$BACKUP_DIR/"

# Backup logs (last 7 days)
find ./logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;

# Backup environment (without sensitive data)
grep -v -E "(API_KEY|PRIVATE_KEY|PASSWORD)" .env > "$BACKUP_DIR/env_backup.txt"

echo "Backup created in $BACKUP_DIR"
EOF
    
    chmod +x "$SCRIPT_DIR/backup.sh"
    
    log "✅ Backup script created"
}

setup_log_rotation() {
    if [ "$ENVIRONMENT" = "production" ]; then
        log "📋 Setting up log rotation..."
        
        sudo tee /etc/logrotate.d/solana-trading-bot > /dev/null << EOF
$SCRIPT_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(whoami)
    postrotate
        docker-compose restart trading-bot
    endscript
}
EOF
        
        log "✅ Log rotation configured"
    fi
}

main() {
    print_banner
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    
    check_prerequisites
    setup_directories
    create_monitoring_config
    create_systemd_service
    deploy_application
    create_backup_script
    setup_log_rotation
    
    show_status
    
    log "🎉 Deployment completed successfully!"
    log "🚀 Your Solana Trading Bot is now running in $MODE mode"
    
    if [ "$MODE" = "simulation" ]; then
        warning "Running in SIMULATION mode - no real trades will be executed"
        info "To switch to live trading, set SIMULATION_MODE=false in .env and redeploy"
    else
        warning "Running in LIVE mode - real trades will be executed!"
        info "Monitor your bot carefully and ensure adequate risk management"
    fi
}

# Run main function
main "$@" 