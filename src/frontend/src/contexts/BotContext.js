import React, { createContext, useContext, useState, useCallback } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

const BotContext = createContext(null);

export const BotProvider = ({ children }) => {
  const [botStatus, setBotStatus] = useState('stopped');
  const queryClient = useQueryClient();

  // Fetch bot status
  const { data: status } = useQuery({
    queryKey: ['botStatus'],
    queryFn: async () => {
      const response = await axios.get('/api/bot/status');
      setBotStatus(response.data.status);
      return response.data;
    },
    refetchInterval: 5000, // Poll every 5 seconds
  });

  // Start bot mutation
  const startBotMutation = useMutation({
    mutationFn: async () => {
      const response = await axios.post('/api/bot/start');
      return response.data;
    },
    onSuccess: () => {
      setBotStatus('running');
      toast.success('Bot started successfully');
      queryClient.invalidateQueries(['botStatus']);
    },
    onError: (error) => {
      toast.error(error.response?.data?.message || 'Failed to start bot');
    },
  });

  // Stop bot mutation
  const stopBotMutation = useMutation({
    mutationFn: async () => {
      const response = await axios.post('/api/bot/stop');
      return response.data;
    },
    onSuccess: () => {
      setBotStatus('stopped');
      toast.success('Bot stopped successfully');
      queryClient.invalidateQueries(['botStatus']);
    },
    onError: (error) => {
      toast.error(error.response?.data?.message || 'Failed to stop bot');
    },
  });

  // Get bot metrics
  const { data: metrics } = useQuery({
    queryKey: ['botMetrics'],
    queryFn: async () => {
      const response = await axios.get('/api/bot/metrics');
      return response.data;
    },
    refetchInterval: 10000, // Poll every 10 seconds
  });

  const startBot = useCallback(() => {
    startBotMutation.mutate();
  }, [startBotMutation]);

  const stopBot = useCallback(() => {
    stopBotMutation.mutate();
  }, [stopBotMutation]);

  return (
    <BotContext.Provider
      value={{
        botStatus,
        metrics,
        startBot,
        stopBot,
        isLoading: startBotMutation.isPending || stopBotMutation.isPending,
      }}
    >
      {children}
    </BotContext.Provider>
  );
};

export const useBot = () => {
  const context = useContext(BotContext);
  if (!context) {
    throw new Error('useBot must be used within a BotProvider');
  }
  return context;
}; 