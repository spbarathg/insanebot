import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Card,
  CardContent,
  CardHeader,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useBot } from '../contexts/BotContext';
import { useAuth } from '../contexts/AuthContext';
import TradingChart from './TradingChart';
import MetricsCard from './MetricsCard';
import RecentTrades from './RecentTrades';

function Dashboard() {
  const { user } = useAuth();
  const { botStatus, metrics, startBot, stopBot, isLoading } = useBot();

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        {/* Header */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h4">
                {getGreeting()}, {user?.username}
              </Typography>
              <Button
                variant="contained"
                color={botStatus === 'running' ? 'error' : 'success'}
                startIcon={botStatus === 'running' ? <StopIcon /> : <PlayIcon />}
                onClick={botStatus === 'running' ? stopBot : startBot}
                disabled={isLoading}
              >
                {isLoading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : botStatus === 'running' ? (
                  'Stop Bot'
                ) : (
                  'Start Bot'
                )}
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Trading Chart */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <TradingChart />
          </Paper>
        </Grid>

        {/* Metrics */}
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <MetricsCard
                title="Performance"
                value={metrics?.performance || '0%'}
                change={metrics?.performanceChange || 0}
              />
            </Grid>
            <Grid item xs={12}>
              <MetricsCard
                title="Total Trades"
                value={metrics?.totalTrades || 0}
                change={metrics?.tradesChange || 0}
              />
            </Grid>
            <Grid item xs={12}>
              <MetricsCard
                title="Success Rate"
                value={metrics?.successRate || '0%'}
                change={metrics?.successRateChange || 0}
              />
            </Grid>
          </Grid>
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Recent Trades</Typography>
              <Button
                startIcon={<RefreshIcon />}
                onClick={() => {/* Refresh trades */}}
              >
                Refresh
              </Button>
            </Box>
            <RecentTrades />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard; 