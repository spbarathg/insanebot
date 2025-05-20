import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';

function MetricsCard({ title, value, change }) {
  const isPositive = change >= 0;
  const changeColor = isPositive ? 'success.main' : 'error.main';
  const ChangeIcon = isPositive ? TrendingUpIcon : TrendingDownIcon;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          {title}
        </Typography>
        <Typography variant="h4" component="div" sx={{ mb: 1 }}>
          {value}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <ChangeIcon sx={{ color: changeColor, mr: 0.5 }} />
          <Typography
            variant="body2"
            sx={{
              color: changeColor,
              fontWeight: 'bold',
            }}
          >
            {Math.abs(change)}%
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}

export default MetricsCard; 