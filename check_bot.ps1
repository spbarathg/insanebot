# Bot status checker script
$server = "user@your-server-ip"

Write-Host "Checking bot status..." -ForegroundColor Cyan

# Execute commands directly for the correct container name
$cmd = "ssh -o ConnectTimeout=5 $server 'docker ps | grep insanebot_trading-bot; echo; docker logs insanebot_trading-bot --tail 10; echo; cat ~/insanebot/.env'"
Invoke-Expression $cmd 