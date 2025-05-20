import React from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Switch,
  FormControlLabel,
  Divider,
} from '@mui/material';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import { toast } from 'react-toastify';
import axios from 'axios';

const validationSchema = Yup.object({
  maxTradeAmount: Yup.number()
    .required('Required')
    .min(0.1, 'Must be at least 0.1')
    .max(1000, 'Must be at most 1000'),
  stopLossPercentage: Yup.number()
    .required('Required')
    .min(1, 'Must be at least 1%')
    .max(50, 'Must be at most 50%'),
  takeProfitPercentage: Yup.number()
    .required('Required')
    .min(1, 'Must be at least 1%')
    .max(100, 'Must be at most 100%'),
  maxConcurrentTrades: Yup.number()
    .required('Required')
    .min(1, 'Must be at least 1')
    .max(10, 'Must be at most 10'),
});

function Settings() {
  const formik = useFormik({
    initialValues: {
      maxTradeAmount: 1.0,
      stopLossPercentage: 5,
      takeProfitPercentage: 10,
      maxConcurrentTrades: 3,
      autoTrading: true,
      notifications: true,
    },
    validationSchema,
    onSubmit: async (values) => {
      try {
        await axios.post('/api/bot/settings', values);
        toast.success('Settings saved successfully');
      } catch (error) {
        toast.error(error.response?.data?.message || 'Failed to save settings');
      }
    },
  });

  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Bot Settings
        </Typography>
        <Divider sx={{ mb: 3 }} />
        
        <form onSubmit={formik.handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                id="maxTradeAmount"
                name="maxTradeAmount"
                label="Maximum Trade Amount (SOL)"
                type="number"
                value={formik.values.maxTradeAmount}
                onChange={formik.handleChange}
                error={formik.touched.maxTradeAmount && Boolean(formik.errors.maxTradeAmount)}
                helperText={formik.touched.maxTradeAmount && formik.errors.maxTradeAmount}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                id="stopLossPercentage"
                name="stopLossPercentage"
                label="Stop Loss Percentage"
                type="number"
                value={formik.values.stopLossPercentage}
                onChange={formik.handleChange}
                error={formik.touched.stopLossPercentage && Boolean(formik.errors.stopLossPercentage)}
                helperText={formik.touched.stopLossPercentage && formik.errors.stopLossPercentage}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                id="takeProfitPercentage"
                name="takeProfitPercentage"
                label="Take Profit Percentage"
                type="number"
                value={formik.values.takeProfitPercentage}
                onChange={formik.handleChange}
                error={formik.touched.takeProfitPercentage && Boolean(formik.errors.takeProfitPercentage)}
                helperText={formik.touched.takeProfitPercentage && formik.errors.takeProfitPercentage}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                id="maxConcurrentTrades"
                name="maxConcurrentTrades"
                label="Maximum Concurrent Trades"
                type="number"
                value={formik.values.maxConcurrentTrades}
                onChange={formik.handleChange}
                error={formik.touched.maxConcurrentTrades && Boolean(formik.errors.maxConcurrentTrades)}
                helperText={formik.touched.maxConcurrentTrades && formik.errors.maxConcurrentTrades}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formik.values.autoTrading}
                    onChange={formik.handleChange}
                    name="autoTrading"
                  />
                }
                label="Auto Trading"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formik.values.notifications}
                    onChange={formik.handleChange}
                    name="notifications"
                  />
                }
                label="Enable Notifications"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                size="large"
                disabled={formik.isSubmitting}
              >
                Save Settings
              </Button>
            </Grid>
          </Grid>
        </form>
      </Paper>
    </Box>
  );
}

export default Settings; 