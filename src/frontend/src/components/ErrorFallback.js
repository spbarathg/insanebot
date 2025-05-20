import React from 'react';
import { Box, Button, Typography, Paper } from '@mui/material';
import { Error as ErrorIcon } from '@mui/icons-material';

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        p: 3,
      }}
    >
      <Paper
        elevation={3}
        sx={{
          p: 4,
          maxWidth: 500,
          textAlign: 'center',
        }}
      >
        <ErrorIcon color="error" sx={{ fontSize: 60, mb: 2 }} />
        <Typography variant="h5" component="h1" gutterBottom>
          Oops! Something went wrong
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          {error.message || 'An unexpected error occurred'}
        </Typography>
        <Box sx={{ mt: 3 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={resetErrorBoundary}
            sx={{ mr: 2 }}
          >
            Try again
          </Button>
          <Button
            variant="outlined"
            color="primary"
            onClick={() => window.location.reload()}
          >
            Refresh page
          </Button>
        </Box>
      </Paper>
    </Box>
  );
}

export default ErrorFallback; 