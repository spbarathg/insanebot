import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

function LoadingSpinner({ message = 'Loading...' }) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
      }}
    >
      <CircularProgress size={60} thickness={4} />
      <Typography
        variant="h6"
        sx={{
          mt: 2,
          color: 'text.secondary',
        }}
      >
        {message}
      </Typography>
    </Box>
  );
}

export default LoadingSpinner; 