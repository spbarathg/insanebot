#!/bin/bash
# ===============================================================================
# 🛡️ PRODUCTION BACKUP SCRIPT - SOLANA TRADING BOT
# ===============================================================================
# Comprehensive backup solution for production trading bot:
# - Database backups (PostgreSQL)
# - Configuration backups
# - Log file archival
# - Trading data backup
# - Encrypted storage
# - Cloud upload (optional)
# ===============================================================================

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_ROOT/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="trading_bot_backup_${DATE}"
RETENTION_DAYS=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Create backup directory structure
setup_backup_directories() {
    log "📁 Setting up backup directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME/database"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME/config"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME/logs"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME/data"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME/monitoring"
    
    log "✅ Backup directories created: $BACKUP_DIR/$BACKUP_NAME"
}

# Backup PostgreSQL database
backup_database() {
    log "🗄️ Backing up PostgreSQL database..."
    
    if [ -z "$DATABASE_URL" ]; then
        warning "DATABASE_URL not configured, skipping database backup"
        return
    fi
    
    # Extract database connection details
    DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    DB_NAME=$(echo "$DATABASE_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')
    DB_USER=$(echo "$DATABASE_URL" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
    
    # Set password from environment
    export PGPASSWORD="$DB_PASSWORD"
    
    # Create database dump
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --verbose --clean --no-owner --no-privileges \
        > "$BACKUP_DIR/$BACKUP_NAME/database/trading_bot_${DATE}.sql"
    
    # Create schema-only backup
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --schema-only --verbose --clean --no-owner --no-privileges \
        > "$BACKUP_DIR/$BACKUP_NAME/database/schema_${DATE}.sql"
    
    # Create data-only backup
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --data-only --verbose --no-owner --no-privileges \
        > "$BACKUP_DIR/$BACKUP_NAME/database/data_${DATE}.sql"
    
    # Compress database backups
    gzip "$BACKUP_DIR/$BACKUP_NAME/database/"*.sql
    
    log "✅ Database backup completed"
}

# Backup configuration files
backup_configuration() {
    log "⚙️ Backing up configuration files..."
    
    # Environment files (excluding secrets)
    if [ -f "$PROJECT_ROOT/env.template" ]; then
        cp "$PROJECT_ROOT/env.template" "$BACKUP_DIR/$BACKUP_NAME/config/"
    fi
    
    # Configuration directories
    if [ -d "$PROJECT_ROOT/config" ]; then
        cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/$BACKUP_NAME/"
    fi
    
    # Docker configuration
    if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
        cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/$BACKUP_NAME/config/"
    fi
    
    if [ -f "$PROJECT_ROOT/docker-compose.prod.yml" ]; then
        cp "$PROJECT_ROOT/docker-compose.prod.yml" "$BACKUP_DIR/$BACKUP_NAME/config/"
    fi
    
    if [ -f "$PROJECT_ROOT/Dockerfile" ]; then
        cp "$PROJECT_ROOT/Dockerfile" "$BACKUP_DIR/$BACKUP_NAME/config/"
    fi
    
    # Monitoring configuration
    if [ -d "$PROJECT_ROOT/monitoring" ]; then
        cp -r "$PROJECT_ROOT/monitoring" "$BACKUP_DIR/$BACKUP_NAME/"
    fi
    
    # Requirements and dependencies
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        cp "$PROJECT_ROOT/requirements.txt" "$BACKUP_DIR/$BACKUP_NAME/config/"
    fi
    
    log "✅ Configuration backup completed"
}

# Backup trading data
backup_trading_data() {
    log "💰 Backing up trading data..."
    
    # Data directory
    if [ -d "$PROJECT_ROOT/data" ]; then
        cp -r "$PROJECT_ROOT/data"/* "$BACKUP_DIR/$BACKUP_NAME/data/" 2>/dev/null || true
    fi
    
    # Portfolio data
    if [ -f "$PROJECT_ROOT/portfolio.json" ]; then
        cp "$PROJECT_ROOT/portfolio.json" "$BACKUP_DIR/$BACKUP_NAME/data/"
    fi
    
    # Create trading summary
    cat > "$BACKUP_DIR/$BACKUP_NAME/data/backup_summary.json" << EOF
{
    "backup_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "${ENVIRONMENT:-unknown}",
    "simulation_mode": "${SIMULATION_MODE:-true}",
    "initial_capital": "${INITIAL_CAPITAL:-0}",
    "backup_type": "automated",
    "retention_days": $RETENTION_DAYS
}
EOF
    
    log "✅ Trading data backup completed"
}

# Backup and archive logs
backup_logs() {
    log "📋 Backing up and archiving logs..."
    
    # Copy recent logs
    if [ -d "$PROJECT_ROOT/logs" ]; then
        # Copy logs from last 7 days
        find "$PROJECT_ROOT/logs" -type f -mtime -7 -name "*.log" -exec cp {} "$BACKUP_DIR/$BACKUP_NAME/logs/" \;
        find "$PROJECT_ROOT/logs" -type f -mtime -7 -name "*.json" -exec cp {} "$BACKUP_DIR/$BACKUP_NAME/logs/" \;
    fi
    
    # Compress old logs
    find "$BACKUP_DIR/$BACKUP_NAME/logs" -type f -name "*.log" -exec gzip {} \;
    
    log "✅ Log backup completed"
}

# Create backup metadata
create_backup_metadata() {
    log "📋 Creating backup metadata..."
    
    # System information
    cat > "$BACKUP_DIR/$BACKUP_NAME/backup_info.json" << EOF
{
    "backup_id": "$BACKUP_NAME",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "system": {
        "os": "$(uname -s)",
        "kernel": "$(uname -r)",
        "architecture": "$(uname -m)"
    },
    "environment": {
        "trading_mode": "${SIMULATION_MODE:-unknown}",
        "environment": "${ENVIRONMENT:-unknown}",
        "version": "${APP_VERSION:-unknown}"
    },
    "backup_contents": {
        "database": true,
        "configuration": true,
        "trading_data": true,
        "logs": true,
        "monitoring": true
    },
    "retention": {
        "retention_days": $RETENTION_DAYS,
        "expire_date": "$(date -d "+$RETENTION_DAYS days" -u +%Y-%m-%dT%H:%M:%SZ)"
    }
}
EOF
    
    # Create checksums
    cd "$BACKUP_DIR/$BACKUP_NAME"
    find . -type f -exec sha256sum {} \; > checksums.sha256
    cd - > /dev/null
    
    log "✅ Backup metadata created"
}

# Encrypt backup (if encryption key is available)
encrypt_backup() {
    log "🔐 Encrypting backup..."
    
    if [ -z "$BACKUP_ENCRYPTION_KEY" ]; then
        warning "BACKUP_ENCRYPTION_KEY not set, skipping encryption"
        return
    fi
    
    # Create encrypted archive
    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    
    # Encrypt the archive
    openssl enc -aes-256-cbc -salt -in "${BACKUP_NAME}.tar.gz" \
        -out "${BACKUP_NAME}.tar.gz.enc" -pass pass:"$BACKUP_ENCRYPTION_KEY"
    
    # Remove unencrypted files
    rm -rf "$BACKUP_NAME"
    rm "${BACKUP_NAME}.tar.gz"
    
    cd - > /dev/null
    
    log "✅ Backup encrypted: ${BACKUP_NAME}.tar.gz.enc"
}

# Upload to cloud storage (optional)
upload_to_cloud() {
    log "☁️ Uploading backup to cloud storage..."
    
    if [ -z "$AWS_S3_BUCKET" ] && [ -z "$BACKUP_CLOUD_URL" ]; then
        info "No cloud storage configured, skipping upload"
        return
    fi
    
    # AWS S3 upload
    if [ -n "$AWS_S3_BUCKET" ] && command -v aws >/dev/null 2>&1; then
        BACKUP_FILE="$BACKUP_DIR/${BACKUP_NAME}.tar.gz.enc"
        if [ ! -f "$BACKUP_FILE" ]; then
            BACKUP_FILE="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
        fi
        
        if [ -f "$BACKUP_FILE" ]; then
            aws s3 cp "$BACKUP_FILE" "s3://$AWS_S3_BUCKET/backups/$(basename "$BACKUP_FILE")"
            log "✅ Backup uploaded to S3: s3://$AWS_S3_BUCKET/backups/"
        fi
    fi
    
    # Generic cloud upload (implement as needed)
    if [ -n "$BACKUP_CLOUD_URL" ]; then
        info "Custom cloud upload not implemented"
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "🧹 Cleaning up old backups..."
    
    # Remove backups older than retention period
    find "$BACKUP_DIR" -name "trading_bot_backup_*" -type f -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "trading_bot_backup_*" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
    
    # Remove empty directories
    find "$BACKUP_DIR" -type d -empty -delete 2>/dev/null || true
    
    log "✅ Old backups cleaned up (retention: $RETENTION_DAYS days)"
}

# Send backup notification
send_notification() {
    local status=$1
    local message=$2
    
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        curl -H "Content-Type: application/json" \
             -X POST \
             -d "{\"content\": \"🛡️ **Backup $status**: $message\\n📅 Timestamp: $(date)\\n🏷️ Backup ID: $BACKUP_NAME\"}" \
             "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
}

# Verify backup integrity
verify_backup() {
    log "🔍 Verifying backup integrity..."
    
    # Check if backup directory exists and has content
    if [ ! -d "$BACKUP_DIR/$BACKUP_NAME" ] && [ ! -f "$BACKUP_DIR/${BACKUP_NAME}.tar.gz.enc" ] && [ ! -f "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" ]; then
        error "Backup verification failed: No backup found"
    fi
    
    # Verify checksums if available
    if [ -f "$BACKUP_DIR/$BACKUP_NAME/checksums.sha256" ]; then
        cd "$BACKUP_DIR/$BACKUP_NAME"
        if sha256sum -c checksums.sha256 >/dev/null 2>&1; then
            log "✅ Backup integrity verified"
        else
            error "Backup integrity check failed"
        fi
        cd - > /dev/null
    fi
}

# Main backup execution
main() {
    log "🚀 Starting Production Backup Process..."
    log "📅 Backup ID: $BACKUP_NAME"
    log "📁 Backup Directory: $BACKUP_DIR"
    
    # Execute backup steps
    setup_backup_directories
    backup_database
    backup_configuration
    backup_trading_data
    backup_logs
    create_backup_metadata
    
    # Verify backup before encryption
    verify_backup
    
    # Optional steps
    encrypt_backup
    upload_to_cloud
    cleanup_old_backups
    
    # Calculate backup size
    if [ -f "$BACKUP_DIR/${BACKUP_NAME}.tar.gz.enc" ]; then
        BACKUP_SIZE=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz.enc" | cut -f1)
    elif [ -f "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" ]; then
        BACKUP_SIZE=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)
    else
        BACKUP_SIZE=$(du -sh "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)
    fi
    
    log "🎉 Backup completed successfully!"
    log "📊 Backup size: $BACKUP_SIZE"
    log "🆔 Backup ID: $BACKUP_NAME"
    
    # Send success notification
    send_notification "Success" "Backup completed successfully (Size: $BACKUP_SIZE)"
}

# Error handling
trap 'error "Backup failed at line $LINENO"; send_notification "Failed" "Backup failed at line $LINENO"' ERR

# Run backup if script is executed directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi 