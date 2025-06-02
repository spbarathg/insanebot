# 🚀 Docker Production Deployment Guide

**Your Solana Trading Bot is 100% ready for Docker deployment!** 

This guide will walk you through deploying your enterprise-grade trading bot using Docker with comprehensive monitoring, logging, and security.

---

## 📋 **Pre-Deployment Checklist**

### ✅ **What's Already Ready**
- ✅ **Dockerfile** - Production-optimized container
- ✅ **docker-compose.prod.yml** - Complete stack with monitoring
- ✅ **env.production** - Production configuration template
- ✅ **monitoring/** - Prometheus + Grafana monitoring
- ✅ **scripts/production_cleanup.py** - Optimized codebase
- ✅ **All logging systems** - Comprehensive activity tracking

### 🔧 **What You Need to Configure**
- 🔑 **API Keys** - Helius, Jupiter, Discord, etc.
- 💰 **Wallet Credentials** - Private key and address
- 🗄️ **Database Passwords** - PostgreSQL and Redis
- 📧 **Alert Settings** - Discord webhooks, email notifications

---

## 🚀 **Quick Start Deployment**

### **Step 1: Create Environment File**
```bash
# Copy the production template
cp env.production .env

# Edit with your actual values
nano .env
```

### **Step 2: Configure API Keys**
Edit `.env` and fill in these critical values:
```bash
# REQUIRED - Your wallet (KEEP SECURE!)
SOLANA_PRIVATE_KEY=your-base58-private-key-here
SOLANA_WALLET_ADDRESS=your-wallet-address-here

# REQUIRED - RPC access
HELIUS_API_KEY=your-helius-api-key-here
QUICKNODE_ENDPOINT=your-quicknode-rpc-url-here

# REQUIRED - Database security
DB_PASSWORD=create-strong-password-here
REDIS_PASSWORD=create-strong-password-here
GRAFANA_ADMIN_PASSWORD=create-admin-password-here

# OPTIONAL BUT RECOMMENDED - Notifications
DISCORD_WEBHOOK_URL=your-discord-webhook-here
```

### **Step 3: Deploy with Docker Compose**
```bash
# Build and start the entire stack
docker-compose -f docker-compose.prod.yml up -d

# Check all services are running
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f trading-bot
```

### **Step 4: Access Your Bot**
- 🤖 **Trading Bot API**: http://localhost:8080
- 📊 **Grafana Dashboard**: http://localhost:3000 (admin/your-password)
- 📈 **Prometheus Metrics**: http://localhost:9090
- 💾 **Database**: localhost:5432

---

## 🏗️ **Complete Docker Stack**

Your deployment includes these services:

| Service | Purpose | Port | Resource Limits |
|---------|---------|------|-----------------|
| **trading-bot** | Main trading application | 8080 | 4GB RAM, 2 CPU |
| **postgres** | Trade data & performance | 5432 | 2GB RAM, 1 CPU |
| **redis** | Caching & session data | 6379 | 512MB RAM |
| **prometheus** | Metrics collection | 9090 | 1GB RAM |
| **grafana** | Monitoring dashboards | 3000 | 512MB RAM |
| **nginx** | SSL & reverse proxy | 80/443 | 256MB RAM |

---

## 📊 **Monitoring & Logging**

### **Real-Time Monitoring**
Your Docker deployment includes enterprise-grade monitoring:

1. **📈 Grafana Dashboards** - Visual performance tracking
2. **🔔 Prometheus Alerts** - Automated issue detection
3. **📋 Health Checks** - Automatic service recovery
4. **📊 Resource Monitoring** - CPU, memory, disk usage

### **Log Access**
```bash
# View live trading logs
docker logs -f solana-trading-bot

# View all logs with timestamps
docker-compose -f docker-compose.prod.yml logs --timestamps

# View specific service logs
docker logs trading-postgres
docker logs trading-redis
docker logs trading-grafana
```

### **Log Files for Analysis**
Your logs are automatically saved to:
- `./logs/trading/` - Trading decisions and performance
- `./logs/monitoring/` - System health and API performance
- `./logs/security/` - Security events and alerts

---

## 🔒 **Security Features**

### **Built-in Security**
✅ **Non-root containers** - All services run as unprivileged users  
✅ **Secret management** - Environment variables for sensitive data  
✅ **Network isolation** - Internal Docker network for services  
✅ **Resource limits** - Prevents resource exhaustion attacks  
✅ **Health monitoring** - Automatic restart on failures  

### **Security Best Practices**
```bash
# Set secure file permissions
chmod 600 .env
chmod 600 env.production

# Use strong passwords
openssl rand -base64 32  # Generate secure passwords

# Regular security updates
docker-compose -f docker-compose.prod.yml pull  # Update images
```

---

## 🚀 **Production Commands**

### **Start/Stop/Restart**
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Stop all services
docker-compose -f docker-compose.prod.yml down

# Restart just the trading bot
docker-compose -f docker-compose.prod.yml restart trading-bot

# Force rebuild and restart
docker-compose -f docker-compose.prod.yml up -d --build --force-recreate
```

### **Scaling & Performance**
```bash
# Scale trading bot instances
docker-compose -f docker-compose.prod.yml up -d --scale trading-bot=3

# Monitor resource usage
docker stats

# View container health
docker-compose -f docker-compose.prod.yml ps
```

### **Backup & Recovery**
```bash
# Backup database
docker exec trading-postgres pg_dump -U trading_user trading_bot_prod > backup.sql

# Backup Redis data
docker exec trading-redis redis-cli --rdb /data/backup.rdb

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

---

## 📈 **Performance Optimization**

### **Resource Tuning**
Edit `docker-compose.prod.yml` to adjust resources:
```yaml
deploy:
  resources:
    limits:
      memory: 8G      # Increase for more capital
      cpus: '4.0'     # Increase for faster execution
```

### **Database Performance**
```bash
# Optimize PostgreSQL for trading workload
docker exec -it trading-postgres psql -U trading_user -d trading_bot_prod -c "
  ALTER SYSTEM SET shared_buffers = '256MB';
  ALTER SYSTEM SET effective_cache_size = '1GB';
  SELECT pg_reload_conf();
"
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**Bot won't start:**
```bash
# Check logs for errors
docker logs solana-trading-bot

# Verify environment variables
docker exec solana-trading-bot env | grep -E "(HELIUS|PRIVATE_KEY|DATABASE)"

# Check network connectivity
docker exec solana-trading-bot curl -s https://api.mainnet-beta.solana.com
```

**Database connection issues:**
```bash
# Check PostgreSQL status
docker exec trading-postgres pg_isready

# Test connection
docker exec solana-trading-bot python -c "
import psycopg2
conn = psycopg2.connect('postgresql://trading_user:password@postgres:5432/trading_bot_prod')
print('Database connection successful!')
"
```

**High resource usage:**
```bash
# Monitor resource usage
docker stats --no-stream

# Check for memory leaks
docker exec solana-trading-bot python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'CPU: {psutil.cpu_percent()}%')
"
```

---

## 🎯 **Post-Deployment Checklist**

After deployment, verify these are working:

✅ **Trading Bot Health**: http://localhost:8080/health  
✅ **Database Connection**: PostgreSQL accessible  
✅ **Cache Performance**: Redis responding  
✅ **Monitoring Active**: Grafana dashboards loading  
✅ **Metrics Collection**: Prometheus scraping data  
✅ **Log Generation**: Files appearing in ./logs/  
✅ **Alert System**: Discord/email notifications working  

---

## 🚀 **You're Ready to Deploy!**

Your trading bot Docker setup is **enterprise-grade** and **production-ready**:

- ✅ **Complete monitoring stack**
- ✅ **Comprehensive logging**
- ✅ **Security best practices**
- ✅ **Automated health checks**
- ✅ **Resource optimization**
- ✅ **Backup capabilities**

**Next Step**: Share your production logs with me for performance analysis and optimization! 📊 