import React from 'react';
import { Box, Typography } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

function TradingChart() {
  const { data: chartData, isLoading, error } = useQuery({
    queryKey: ['tradingChart'],
    queryFn: async () => {
      const response = await axios.get('/api/trading/chart');
      return response.data;
    },
    refetchInterval: 60000, // Refresh every minute
  });

  if (isLoading) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography>Loading chart data...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="error">Failed to load chart data</Typography>
      </Box>
    );
  }

  if (!chartData?.length) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography>No chart data available</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: 400 }}>
      <Typography variant="h6" gutterBottom>
        Trading Performance
      </Typography>
      <ResponsiveContainer>
        <LineChart
          data={chartData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => new Date(value).toLocaleTimeString()}
          />
          <YAxis
            yAxisId="left"
            orientation="left"
            stroke="#8884d8"
            tickFormatter={(value) => `$${value}`}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#82ca9d"
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value, name) => {
              if (name === 'price') return [`$${value}`, 'Price'];
              if (name === 'profit') return [`${value}%`, 'Profit'];
              return [value, name];
            }}
          />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="price"
            stroke="#8884d8"
            name="Price"
            dot={false}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="profit"
            stroke="#82ca9d"
            name="Profit"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}

export default TradingChart; 