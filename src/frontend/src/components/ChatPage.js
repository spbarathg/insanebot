import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, TextField, IconButton, CircularProgress, Fade } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import axios from 'axios';

const CHAT_MAX_WIDTH = 720;
const HISTORY_KEY = 'chat_history_v1';

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Load history on mount
  useEffect(() => {
    const saved = localStorage.getItem(HISTORY_KEY);
    if (saved) {
      setMessages(JSON.parse(saved));
    } else {
      setMessages([
        { role: 'system', content: 'Welcome! Ask me anything about your trading bot performance.', timestamp: Date.now() }
      ]);
    }
  }, []);

  // Save history on change
  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(messages));
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Poll for new bot messages
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await axios.get('/messages/poll');
        if (res.data && Array.isArray(res.data)) {
          // Only add messages that are new
          const lastTs = messages.length ? messages[messages.length - 1].timestamp : 0;
          const newMsgs = res.data.filter(m => m.timestamp > lastTs);
          if (newMsgs.length) setMessages(prev => [...prev, ...newMsgs]);
        }
      } catch {}
    }, 5000);
    return () => clearInterval(poll);
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMessage = { role: 'user', content: input, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    try {
      const res = await axios.post('/chat', { message: userMessage.content });
      setMessages((prev) => [...prev, { role: 'ai', content: res.data.response, timestamp: Date.now() }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'ai', content: 'Error: Could not get response.', timestamp: Date.now() }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#181a20', display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100vw', position: 'relative' }}>
      <Box sx={{ mt: 8, mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" sx={{ color: '#fff', fontWeight: 700, letterSpacing: -1, mb: 1, fontSize: { xs: 32, sm: 40 } }}>
          Trading Bot Chat
        </Typography>
        <Typography variant="subtitle1" sx={{ color: '#b1b5c3', fontWeight: 400, fontSize: 18, letterSpacing: 0.2 }}>
          Your AI-powered trading assistant
        </Typography>
      </Box>
      <Box
        sx={{
          width: '100%',
          maxWidth: CHAT_MAX_WIDTH,
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          mx: 'auto',
          pb: 12, // leave space for the fixed input bar
        }}
      >
        <Box sx={{ width: '100%', maxWidth: CHAT_MAX_WIDTH, display: 'flex', flexDirection: 'column', gap: 0, mt: 4 }}>
          {messages.map((msg, idx) => (
            <Fade in key={idx} timeout={400}>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  width: '100%',
                  mb: 2,
                }}
              >
                <Box
                  sx={{
                    px: 2.5,
                    py: 1.5,
                    borderRadius: 3,
                    bgcolor: msg.role === 'user' ? 'linear-gradient(90deg, #3772ff 60%, #5f7cff 100%)' : '#23242b',
                    color: '#fff',
                    fontSize: 18,
                    fontWeight: 500,
                    maxWidth: '60%',
                    minWidth: 80,
                    wordBreak: 'break-word',
                    boxShadow: msg.role === 'user' ? '0 2px 8px #3772ff22' : '0 1px 4px #0001',
                    borderTopRightRadius: msg.role === 'user' ? 0 : 12,
                    borderTopLeftRadius: msg.role === 'user' ? 12 : 0,
                    opacity: 0.97,
                    ml: msg.role === 'user' ? 'auto' : 0,
                    mr: msg.role === 'user' ? 0 : 'auto',
                  }}
                >
                  {msg.content}
                  <Typography sx={{ fontSize: 12, color: '#b1b5c3', mt: 0.5, textAlign: msg.role === 'user' ? 'right' : 'left' }}>
                    {formatTime(msg.timestamp)}
                  </Typography>
                </Box>
              </Box>
            </Fade>
          ))}
          <div ref={messagesEndRef} />
        </Box>
      </Box>
      {/* Fixed input bar at the bottom */}
      <Box
        component="form"
        onSubmit={handleSend}
        sx={{
          position: 'fixed',
          left: 0,
          bottom: 0,
          width: '100vw',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          bgcolor: 'transparent',
          zIndex: 10,
          py: 2,
        }}
      >
        <Box
          sx={{
            width: '100%',
            maxWidth: CHAT_MAX_WIDTH,
            display: 'flex',
            alignItems: 'center',
            gap: 1.5,
            bgcolor: 'rgba(35,36,43,0.98)',
            borderRadius: 50,
            boxShadow: '0 2px 16px #0002',
            px: 2,
            py: 1.2,
          }}
        >
          <TextField
            fullWidth
            variant="standard"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            InputProps={{
              disableUnderline: true,
              style: {
                color: '#fff',
                fontSize: 18,
                paddingLeft: 8,
                paddingRight: 8,
                background: 'none',
              },
            }}
            sx={{
              bgcolor: 'transparent',
              borderRadius: 50,
              mx: 1,
              input: { color: '#fff' },
            }}
            autoFocus
            disabled={loading}
          />
          <IconButton
            type="submit"
            color="primary"
            disabled={loading || !input.trim()}
            sx={{
              bgcolor: '#3772ff',
              color: '#fff',
              borderRadius: '50%',
              width: 48,
              height: 48,
              boxShadow: '0 2px 8px #3772ff33',
              '&:hover': { bgcolor: '#5f7cff' },
              transition: 'background 0.2s',
            }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
          </IconButton>
        </Box>
      </Box>
    </Box>
  );
};

export default ChatPage; 