# 🆘 Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for recovering from various disaster scenarios that may affect the Enhanced Ant Bot trading system.

## 🔄 Recovery Time Objectives (RTO) & Recovery Point Objectives (RPO)

| Scenario | RTO Target | RPO Target | Priority |
|----------|------------|------------|----------|
| Complete System Failure | 15 minutes | 5 minutes | Critical |
| Database Corruption | 30 minutes | 10 minutes | Critical |
| Network Partition | 5 minutes | 1 minute | High |
| Application Bug | 10 minutes | 2 minutes | High |
| External API Outage | 2 minutes | 30 seconds | Medium |

## 📋 Pre-Disaster Checklist

### Daily Preparations
- [ ] Verify backup systems are operational
- [ ] Test failover mechanisms
- [ ] Confirm monitoring alerts are functional
- [ ] Validate disaster recovery contacts are current

### Weekly Preparations
- [ ] Full disaster recovery drill
- [ ] Update runbook procedures
- [ ] Review and test backup restoration
- [ ] Validate off-site backup integrity

## 🚨 Emergency Response Procedures

### Immediate Response (First 5 Minutes)

1. **Assess the Situation**
   ```bash
   # Check system status
   curl -f http://localhost:8080/health || echo "SYSTEM DOWN"
   
   # Check critical components
   docker ps | grep -E "(trading-bot|postgres|redis)"
   
   # Review recent logs
   tail -n 100 logs/system.log | grep -i error
   ```

2. **Stop Trading Activities**
   ```bash
   # Emergency trading halt
   curl -X POST http://localhost:8080/emergency/halt \
     -H "Authorization: Bearer $EMERGENCY_TOKEN"
   
   # Verify halt is effective
   curl http://localhost:8080/status/trading
   ```

3. **Notify Stakeholders**
   ```bash
   # Send emergency notification
   python scripts/emergency_notification.py \
     --severity critical \
     --message "Trading system down - disaster recovery initiated"
   ```

### Detailed Recovery Procedures

## 💾 Scenario 1: Complete System Failure

### Symptoms
- All services unresponsive
- Docker containers crashed
- System monitoring down

### Recovery Steps

1. **Assess Infrastructure**
   ```bash
   # Check Docker daemon
   sudo systemctl status docker
   
   # Check disk space
   df -h
   
   # Check memory
   free -h
   
   # Check system load
   uptime
   ```

2. **Restart Core Infrastructure**
   ```bash
   # Restart Docker if needed
   sudo systemctl restart docker
   
   # Start from clean state
   cd /path/to/enhanced-ant-bot
   docker-compose -f docker-compose.prod.yml down
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Verify Service Recovery**
   ```bash
   # Wait for services to start
   sleep 30
   
   # Check service health
   curl http://localhost:8080/health
   curl http://localhost:3000  # Grafana
   curl http://localhost:9090  # Prometheus
   ```

4. **Restore Trading State**
   ```bash
   # Check portfolio state
   curl http://localhost:8080/portfolio/status
   
   # Verify risk parameters
   curl http://localhost:8080/risk/current
   
   # Resume trading (if safe)
   curl -X POST http://localhost:8080/trading/resume \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

## 🗄️ Scenario 2: Database Corruption/Failure

### Symptoms
- PostgreSQL connection errors
- Data integrity issues
- Backup restoration needed

### Recovery Steps

1. **Stop All Trading**
   ```bash
   curl -X POST http://localhost:8080/emergency/halt
   ```

2. **Assess Database Damage**
   ```bash
   # Connect to database
   docker exec -it trading-postgres psql -U $DB_USER -d trading_bot_prod
   
   # Check database integrity
   \l
   \dt
   SELECT count(*) FROM trades;
   SELECT count(*) FROM portfolio_positions;
   ```

3. **Restore from Backup**
   ```bash
   # Stop database container
   docker stop trading-postgres
   
   # Restore from latest backup
   cd backups
   LATEST_BACKUP=$(ls -t *.sql | head -1)
   
   # Start fresh database
   docker-compose up -d postgres
   
   # Wait for startup
   sleep 10
   
   # Restore data
   docker exec -i trading-postgres psql -U $DB_USER -d trading_bot_prod < $LATEST_BACKUP
   ```

4. **Verify Data Integrity**
   ```bash
   # Check critical tables
   docker exec trading-postgres psql -U $DB_USER -d trading_bot_prod -c "
     SELECT 
       (SELECT count(*) FROM trades) as trade_count,
       (SELECT count(*) FROM portfolio_positions) as position_count,
       (SELECT max(created_at) FROM trades) as latest_trade;
   "
   ```

5. **Resume Operations**
   ```bash
   # Restart trading bot
   docker restart trading-bot
   
   # Verify system health
   curl http://localhost:8080/health
   
   # Resume trading
   curl -X POST http://localhost:8080/trading/resume
   ```

## 🌐 Scenario 3: Network Connectivity Issues

### Symptoms
- External API failures
- Solana RPC timeouts
- Intermittent connectivity

### Recovery Steps

1. **Diagnose Network Issues**
   ```bash
   # Test external connectivity
   ping -c 3 8.8.8.8
   curl -I https://api.helius.xyz
   curl -I https://quote-api.jup.ag
   
   # Check DNS resolution
   nslookup api.helius.xyz
   nslookup quote-api.jup.ag
   ```

2. **Switch to Backup Services**
   ```bash
   # Update configuration for backup RPCs
   curl -X POST http://localhost:8080/config/update \
     -H "Content-Type: application/json" \
     -d '{
       "primary_rpc": "backup_helius_endpoint",
       "backup_rpc": "quicknode_endpoint",
       "failover_enabled": true
     }'
   ```

3. **Monitor Service Recovery**
   ```bash
   # Watch connectivity metrics
   curl http://localhost:8080/metrics | grep -E "(api_availability|rpc_latency)"
   ```

## 💥 Scenario 4: Critical Application Bug

### Symptoms
- Execution errors
- Memory leaks
- Logic errors in trading

### Recovery Steps

1. **Immediate Mitigation**
   ```bash
   # Stop trading immediately
   curl -X POST http://localhost:8080/emergency/halt
   
   # Collect diagnostic information
   curl http://localhost:8080/debug/memory
   curl http://localhost:8080/debug/goroutines
   ```

2. **Rollback to Previous Version**
   ```bash
   # Stop current version
   docker-compose down
   
   # Rollback to last known good image
   export ANTBOT_VERSION=v1.2.3  # Last stable version
   docker-compose up -d
   ```

3. **Verify Rollback Success**
   ```bash
   # Check version
   curl http://localhost:8080/version
   
   # Run health checks
   curl http://localhost:8080/health
   ```

## 🔧 Post-Recovery Procedures

### Immediate Post-Recovery (First Hour)

1. **System Validation**
   ```bash
   # Run comprehensive health check
   python scripts/health_check_comprehensive.py
   
   # Verify all metrics are reporting
   curl http://localhost:9090/api/v1/targets
   ```

2. **Portfolio Reconciliation**
   ```bash
   # Check portfolio consistency
   curl http://localhost:8080/portfolio/reconcile
   
   # Verify position accuracy
   curl http://localhost:8080/portfolio/verify
   ```

3. **Resume Trading (If Safe)**
   ```bash
   # Gradual trading resumption
   curl -X POST http://localhost:8080/trading/resume \
     -d '{"mode": "conservative", "max_position_size": 0.005}'
   ```

### Extended Post-Recovery (First 24 Hours)

1. **Performance Monitoring**
   - Monitor execution latency
   - Check for any anomalies
   - Verify risk management systems

2. **Data Validation**
   ```bash
   # Run data integrity checks
   python scripts/data_integrity_check.py
   
   # Validate trading history
   python scripts/validate_trading_history.py
   ```

3. **Documentation Update**
   - Document what happened
   - Update procedures if needed
   - Conduct post-mortem review

## 📞 Emergency Contacts

| Role | Primary | Secondary | Emergency |
|------|---------|-----------|-----------|
| System Administrator | +1-XXX-XXX-XXXX | +1-XXX-XXX-XXXX | ops@antbot.local |
| Database Administrator | +1-XXX-XXX-XXXX | +1-XXX-XXX-XXXX | dba@antbot.local |
| Security Team | +1-XXX-XXX-XXXX | +1-XXX-XXX-XXXX | security@antbot.local |
| Business Stakeholder | +1-XXX-XXX-XXXX | +1-XXX-XXX-XXXX | business@antbot.local |

## 🧪 Testing Procedures

### Monthly Disaster Recovery Test

```bash
#!/bin/bash
# Monthly DR Test Script

echo "Starting Monthly DR Test..."

# 1. Simulate database failure
docker stop trading-postgres
sleep 30

# 2. Verify alerts triggered
curl http://localhost:9090/api/v1/alerts

# 3. Execute recovery procedure
# ... (follow database recovery steps)

# 4. Validate recovery
curl http://localhost:8080/health

echo "DR Test Complete - Document results"
```

### Automated Recovery Testing

```bash
#!/bin/bash
# Automated Recovery Test

# Test various failure scenarios
python scripts/chaos_testing.py \
  --scenario database_failure \
  --duration 300 \
  --recovery_target 15

python scripts/chaos_testing.py \
  --scenario network_partition \
  --duration 120 \
  --recovery_target 5
```

## 📊 Recovery Metrics

Track these metrics during recovery:

- **Mean Time to Detection (MTTD)**: Time from failure to alert
- **Mean Time to Response (MTTR)**: Time from alert to response start
- **Mean Time to Recovery (MTTR)**: Time from failure to full recovery
- **Data Loss**: Amount of data lost during incident

## 🔄 Continuous Improvement

### After Each Incident

1. **Conduct Post-Mortem**
   - Root cause analysis
   - Timeline review
   - Lessons learned

2. **Update Procedures**
   - Revise runbooks
   - Improve automation
   - Enhance monitoring

3. **Training Updates**
   - Update team training
   - Practice new procedures
   - Validate improvements

## 📚 Additional Resources

- [System Architecture Documentation](../architecture/system_overview.md)
- [Monitoring Runbook](./monitoring_runbook.md)
- [Security Incident Response](./security_incident_response.md)
- [Backup and Recovery Procedures](./backup_recovery.md) 