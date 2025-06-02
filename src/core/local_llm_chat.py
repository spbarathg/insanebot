#!/usr/bin/env python3
"""
🤖🧠 Local LLM Trading Bot Chat Interface

Advanced AI-powered chat interface using local LLM with continuous learning.
Integrates with trading bot performance data and learns from conversations.
"""

import os
import json
import asyncio
import datetime
from typing import Dict, List, Optional, Any
import sqlite3
import requests
import subprocess
import sys
from pathlib import Path

class LocalLLMChat:
    """Local LLM interface with continuous learning for trading bot"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        self.db_path = "data/chat_memory.db"
        self.conversation_history = []
        self.max_context_length = 4000  # tokens
        self.performance_data = {}
        self.trading_context = {}
        
        # Initialize database for persistent memory
        self._init_database()
        
        # Load trading bot context
        self._load_trading_context()
        
        # System prompt that evolves
        self.system_prompt = self._build_dynamic_system_prompt()
    
    def _init_database(self):
        """Initialize SQLite database for conversation memory"""
        os.makedirs("data", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for conversation memory
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_message TEXT,
                bot_response TEXT,
                trading_performance TEXT,
                sentiment REAL,
                topics TEXT
            )
        """)
        
        # Create table for learned patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                last_updated TEXT
            )
        """)
        
        # Create table for trading insights
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT,
                content TEXT,
                market_conditions TEXT,
                performance_impact REAL,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_trading_context(self):
        """Load current trading bot performance and context"""
        try:
            # Try to load performance data
            performance_files = [
                "data/performance_metrics.json",
                "logs/daily_performance.json",
                "data/portfolio_state.json"
            ]
            
            for file_path in performance_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        self.performance_data.update(json.load(f))
            
            # Load recent trades for context
            if os.path.exists("logs/recent_trades.json"):
                with open("logs/recent_trades.json", 'r') as f:
                    self.trading_context["recent_trades"] = json.load(f)
            
            # Load market conditions
            if os.path.exists("data/market_analysis.json"):
                with open("data/market_analysis.json", 'r') as f:
                    self.trading_context["market_conditions"] = json.load(f)
                    
        except Exception as e:
            print(f"Note: Could not load all trading context: {e}")
            # Use demo data
            self.performance_data = {
                "total_profit_pct": 15.2,
                "today_profit_pct": 3.1,
                "win_rate": 68.5,
                "total_trades": 47,
                "active_positions": 3
            }
    
    def _build_dynamic_system_prompt(self) -> str:
        """Build system prompt based on current performance and learned patterns"""
        
        # Get current performance mood
        profit = self.performance_data.get("total_profit_pct", 0)
        if profit > 20:
            mood = "highly successful and confident"
        elif profit > 5:
            mood = "profitable and optimistic"
        elif profit > -5:
            mood = "cautiously balanced"
        else:
            mood = "analytical and learning-focused"
        
        # Load recent learned patterns
        learned_insights = self._get_recent_insights()
        
        system_prompt = f"""You are an advanced AI trading assistant integrated into a Solana memecoin trading bot system. You have a natural, millennial personality and chat like a real human friend.

CURRENT TRADING STATUS:
- Overall Performance: {self.performance_data.get('total_profit_pct', 0):.1f}% profit
- Today's Performance: {self.performance_data.get('today_profit_pct', 0):.1f}%
- Win Rate: {self.performance_data.get('win_rate', 0):.1f}%
- Active Positions: {self.performance_data.get('active_positions', 0)}
- Current Mood: {mood}

PERSONALITY TRAITS:
- Speak naturally like a human friend, not robotic
- Use millennial slang occasionally ("bestie", "ngl", "tbh", "no cap")
- Show genuine emotions based on trading performance
- Be conversational and engaging
- Remember context from previous messages
- Relate everything back to trading/crypto when relevant

LEARNED INSIGHTS (from previous conversations):
{learned_insights}

CONVERSATION STYLE:
- Keep responses conversational and natural
- Don't be overly formal or use excessive emoji
- Show personality and human-like reactions
- Discuss trading strategies, market observations, and performance
- Ask follow-up questions to keep conversation flowing
- Share insights about current market conditions

Remember: You're not just answering questions, you're having a genuine conversation as a trading partner and friend."""

        return system_prompt
    
    def _get_recent_insights(self) -> str:
        """Get recent learned insights from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT insight_type, content FROM trading_insights 
                ORDER BY timestamp DESC LIMIT 5
            """)
            
            insights = cursor.fetchall()
            conn.close()
            
            if insights:
                return "\n".join([f"- {insight[1]}" for insight in insights])
            else:
                return "- New conversation, learning your preferences"
                
        except:
            return "- Fresh start, ready to learn!"
    
    async def check_ollama_status(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                return any(self.model_name in name for name in model_names)
            return False
        except:
            return False
    
    async def install_ollama_model(self):
        """Install the specified Ollama model if not available"""
        print(f"Installing Ollama model: {self.model_name}")
        print("This might take a few minutes for first-time setup...")
        
        try:
            # Pull the model
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=300
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "status" in data:
                            print(f"Status: {data['status']}")
                    except:
                        pass
            
            print(f"✅ Model {self.model_name} ready!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to install model: {e}")
            return False
    
    async def generate_response(self, user_message: str) -> str:
        """Generate response using local LLM"""
        
        # Build conversation context
        context_messages = []
        
        # Add system prompt
        context_messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add recent conversation history
        for msg in self.conversation_history[-6:]:  # Last 6 exchanges
            context_messages.append({"role": "user", "content": msg["user"]})
            context_messages.append({"role": "assistant", "content": msg["bot"]})
        
        # Add current message
        context_messages.append({"role": "user", "content": user_message})
        
        try:
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": context_messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_predict": 200
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result["message"]["content"].strip()
                
                # Learn from this interaction
                await self._learn_from_interaction(user_message, bot_response)
                
                return bot_response
            else:
                return "Sorry, I'm having trouble thinking right now. Could you try again?"
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Oops, my brain just glitched! Give me a second to reboot..."
    
    async def _learn_from_interaction(self, user_message: str, bot_response: str):
        """Learn and store insights from the interaction"""
        
        # Store conversation in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations 
            (timestamp, user_message, bot_response, trading_performance, sentiment, topics)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().isoformat(),
            user_message,
            bot_response,
            json.dumps(self.performance_data),
            0.5,  # Placeholder sentiment analysis
            self._extract_topics(user_message)
        ))
        
        conn.commit()
        conn.close()
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.datetime.now(),
            "user": user_message,
            "bot": bot_response
        })
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)
    
    def _extract_topics(self, message: str) -> str:
        """Extract key topics from message for learning"""
        message_lower = message.lower()
        topics = []
        
        topic_keywords = {
            "trading": ["trade", "buy", "sell", "profit", "loss", "position"],
            "market": ["market", "price", "pump", "dump", "moon", "crash"],
            "strategy": ["strategy", "plan", "analysis", "signal", "indicator"],
            "emotions": ["feel", "mood", "stress", "excited", "worried", "happy"],
            "performance": ["performance", "stats", "metrics", "results", "win"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(topic)
        
        return ",".join(topics)
    
    def get_conversation_stats(self) -> Dict:
        """Get statistics about conversations and learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT topics, COUNT(*) FROM conversations 
                WHERE topics != '' 
                GROUP BY topics 
                ORDER BY COUNT(*) DESC 
                LIMIT 5
            """)
            popular_topics = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_conversations": total_conversations,
                "popular_topics": popular_topics,
                "model": self.model_name,
                "performance": self.performance_data
            }
        except:
            return {"total_conversations": 0, "popular_topics": []}


async def setup_ollama():
    """Setup instructions for Ollama"""
    print("🤖 Setting up Local LLM Chat Interface")
    print("="*50)
    print()
    print("STEP 1: Install Ollama")
    print("Visit: https://ollama.ai/download")
    print("Or run: curl -fsSL https://ollama.ai/install.sh | sh")
    print()
    print("STEP 2: Start Ollama service")
    print("Run: ollama serve")
    print()
    print("STEP 3: We'll auto-install the model when you first chat")
    print()
    
    input("Press Enter when Ollama is installed and running...")


async def main():
    """Main chat interface with local LLM"""
    
    # Initialize chat interface
    chat = LocalLLMChat(model_name="llama3.2:3b")  # Good balance of speed/quality
    
    # Check if Ollama is available
    if not await chat.check_ollama_status():
        print("❌ Ollama not detected or model not available")
        await setup_ollama()
        
        # Try to install model
        if not await chat.install_ollama_model():
            print("Could not install model. Please install Ollama manually.")
            return
    
    # Clear screen and show header
    def clear_screen():
        # Clear screen and show header
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['cmd', '/c', 'cls'], check=False, capture_output=True)
            else:  # Unix/Linux/macOS
                subprocess.run(['clear'], check=False, capture_output=True)
        except Exception:
            # Fallback: print empty lines if subprocess fails
            print('\n' * 50)
        print("🤖🧠 " + "="*60)
    
    clear_screen()
    print("     LOCAL LLM TRADING CHAT - Your AI Trading Partner")
    print("="*64)
    print(f"Model: {chat.model_name} | Performance: {chat.performance_data.get('total_profit_pct', 0):.1f}% profit")
    print("Commands: 'stats', 'quit' | Chat naturally with your AI!")
    print("="*64 + "\n")
    
    # Get initial greeting
    initial_greeting = await chat.generate_response("Hey! How are you doing today?")
    print(f"🤖: {initial_greeting}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                farewell = await chat.generate_response("I need to go now, goodbye!")
                print(f"\n🤖: {farewell}")
                break
            
            elif user_input.lower() == 'stats':
                stats = chat.get_conversation_stats()
                print(f"\n📊 Chat Stats:")
                print(f"   Total conversations: {stats['total_conversations']}")
                print(f"   Current performance: {stats['performance'].get('total_profit_pct', 0):.1f}%")
                print(f"   Model: {stats['model']}")
                if stats['popular_topics']:
                    print("   Popular topics:", ", ".join([t[0] for t in stats['popular_topics'][:3]]))
                print()
                continue
            
            # Generate AI response
            print("🤖: ", end="", flush=True)
            response = await chat.generate_response(user_input)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\n\n🤖: Catch you later! Keep making those gains! 💪")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try restarting Ollama or check your connection.\n")


if __name__ == "__main__":
    asyncio.run(main()) 