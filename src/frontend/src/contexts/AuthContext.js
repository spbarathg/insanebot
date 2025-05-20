import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';
import { jwtDecode } from 'jwt-decode';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';

const AuthContext = createContext(null);

// Test credentials
const TEST_CREDENTIALS = {
  username: 'admin',
  password: 'Admin@123'
};

// Mock JWT token generator
const generateMockToken = (username) => {
  const payload = {
    sub: '1',
    username: username,
    role: 'admin',
    exp: Math.floor(Date.now() / 1000) + (24 * 60 * 60) // 24 hours
  };
  return btoa(JSON.stringify(payload)); // Simple base64 encoding for demo
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const decoded = JSON.parse(atob(token));
        if (decoded.exp * 1000 > Date.now()) {
          setUser(decoded);
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        } else {
          localStorage.removeItem('token');
          delete axios.defaults.headers.common['Authorization'];
        }
      } catch (error) {
        console.error('Token validation failed:', error);
        localStorage.removeItem('token');
      }
    }
    setLoading(false);
  }, []);

  const login = async (credentials) => {
    try {
      // Mock authentication
      if (credentials.username === TEST_CREDENTIALS.username && 
          credentials.password === TEST_CREDENTIALS.password) {
        const token = generateMockToken(credentials.username);
        
        localStorage.setItem('token', token);
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        
        const decoded = JSON.parse(atob(token));
        setUser(decoded);
        
        toast.success('Login successful!');
        navigate('/dashboard');
      } else {
        throw new Error('Invalid credentials');
      }
    } catch (error) {
      toast.error(error.message || 'Login failed');
      throw error;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
    navigate('/login');
    toast.info('Logged out successfully');
  };

  const refreshToken = async () => {
    try {
      const response = await axios.post('/api/auth/refresh');
      const { token } = response.data;
      
      localStorage.setItem('token', token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      const decoded = jwtDecode(token);
      setUser(decoded);
    } catch (error) {
      console.error('Token refresh failed:', error);
      logout();
    }
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, refreshToken }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 