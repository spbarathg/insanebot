import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Box,
  Typography,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

function RecentTrades() {
  const { data: trades, isLoading, error } = useQuery({
    queryKey: ['recentTrades'],
    queryFn: async () => {
      const response = await axios.get('/api/trades/recent');
      return response.data;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  if (isLoading) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography>Loading trades...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="error">Failed to load trades</Typography>
      </Box>
    );
  }

  if (!trades?.length) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography>No recent trades</Typography>
      </Box>
    );
  }

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Time</TableCell>
            <TableCell>Token</TableCell>
            <TableCell>Type</TableCell>
            <TableCell align="right">Amount</TableCell>
            <TableCell align="right">Price</TableCell>
            <TableCell align="right">Total</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {trades.map((trade) => (
            <TableRow key={trade.id}>
              <TableCell>
                {new Date(trade.timestamp).toLocaleString()}
              </TableCell>
              <TableCell>{trade.token}</TableCell>
              <TableCell>
                <Chip
                  label={trade.type}
                  color={trade.type === 'BUY' ? 'success' : 'error'}
                  size="small"
                />
              </TableCell>
              <TableCell align="right">{trade.amount}</TableCell>
              <TableCell align="right">${trade.price}</TableCell>
              <TableCell align="right">${trade.total}</TableCell>
              <TableCell>
                <Chip
                  label={trade.status}
                  color={
                    trade.status === 'COMPLETED'
                      ? 'success'
                      : trade.status === 'PENDING'
                      ? 'warning'
                      : 'error'
                  }
                  size="small"
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default RecentTrades; 