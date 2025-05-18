import React, { useState, useEffect } from 'react';
import './App.css';
import { useAuth } from './contexts/AuthContext';
import { useBot } from './contexts/BotContext';

function App() {
  const { user, login, logout } = useAuth();
  const { botStatus, startBot, stopBot } = useBot();
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (credentials) => {
    setIsLoading(true);
    try {
      await login(credentials);
    } catch (error) {
      console.error('Login failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    setIsLoading(true);
    try {
      await logout();
    } catch (error) {
      console.error('Logout failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Greeting based on time
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  // Minimalistic logout icon SVG
  const logoutIcon = (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ display: 'block' }}
    >
      <path d="M13 16v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V3a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v1" />
      <polyline points="17 8 20 10 17 12" />
      <line x1="20" y1="10" x2="9" y2="10" />
    </svg>
  );

  return (
    <div className="grok-app-bg">
      <header className="grok-header">
        <div></div>
        <div className="grok-profile">
          {user ? (
            <div className="grok-profile-icon">{user.username[0]}</div>
          ) : (
            <div className="login-form">
              <h2>Login</h2>
              <form onSubmit={handleLogin}>
                <input
                  type="text"
                  placeholder="Username"
                  name="username"
                  required
                />
                <input
                  type="password"
                  placeholder="Password"
                  name="password"
                  required
                />
                <button type="submit" disabled={isLoading}>
                  {isLoading ? 'Logging in...' : 'Login'}
                </button>
              </form>
            </div>
          )}
          <button className="grok-login-btn" onClick={handleLogout} disabled={isLoading}>
            {isLoading ? 'Logging out...' : logoutIcon}
          </button>
        </div>
      </header>
      <main className="grok-main">
        {user && (
          <>
            <div className="grok-greeting">
              <span>{getGreeting()}, {user.username}.</span>
              <span className="grok-sub">How can I help you today?</span>
            </div>
            <div className="bot-controls">
              <h2>Trading Bot Status: {botStatus}</h2>
              <button
                onClick={botStatus === 'running' ? stopBot : startBot}
                disabled={isLoading}
              >
                {botStatus === 'running' ? 'Stop Bot' : 'Start Bot'}
              </button>
            </div>
          </>
        )}
        <form className="grok-input-form" onSubmit={(e) => {
          e.preventDefault();
          // Handle chat message
        }}>
          <input
            className="grok-input-box"
            type="text"
            placeholder="Ask me the current status of the trading bot."
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                // Handle chat message
              }
            }}
          />
          <button className="grok-input-btn" type="submit">→</button>
        </form>
      </main>
    </div>
  );
}

export default App;
