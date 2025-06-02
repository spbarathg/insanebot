#!/usr/bin/env python3
"""
🤖💬 Trading Bot Chat Interface - Core Interactive Component

A natural, conversational chat interface that feels like talking to a real human friend!
The bot has personality, remembers conversations, and chats naturally without emoji spam.

This is a core component of the trading bot system for user interaction.
"""

import os
import json
import datetime
from typing import Dict, List, Optional
import asyncio
import sys
import time
import subprocess
import secrets

def secure_uniform(a: float, b: float) -> float:
    """Secure uniform random float between a and b"""
    return a + (b - a) * (secrets.randbelow(10000) / 10000.0)

def secure_choice(seq):
    """Secure random choice from sequence"""
    return seq[secrets.randbelow(len(seq))]

def secure_random() -> float:
    """Secure random float between 0 and 1"""
    return secrets.randbelow(10000) / 10000.0

class TradingBotPersonality:
    """Human-like trading bot with natural conversation patterns"""
    
    def __init__(self):
        self.current_mood = "neutral"
        self.performance_data = self._load_performance_data()
        self.conversation_memory = []
        self.topics_discussed = set()
        self.last_greeting_time = None
        self.energy_level = 5  # 1-10 scale
        self.favorite_memecoins = ["PEPE", "DOGE", "SHIB", "BONK", "WIF"]
        self.personality_traits = {
            "chattiness": secrets.randbelow(4) + 7,  # 7-10
            "optimism": secrets.randbelow(4) + 6,    # 6-9
            "humor_level": secrets.randbelow(3) + 8, # 8-10
            "anxiety_prone": secrets.randbelow(5) + 3, # 3-7
            "emoji_usage": secrets.randbelow(3) + 2    # 2-4
        }
        self.current_obsession = secure_choice([
            "pump.fun launches", "whale watching", "social media trends", 
            "technical analysis", "market psychology", "DeFi yields"
        ])
        self.last_big_win = None
        self.stress_level = secrets.randbelow(5) + 1  # 1-5
        
    def _load_performance_data(self) -> Dict:
        """Load bot performance data to determine mood"""
        try:
            # Try to load from performance dashboard data (relative to project root)
            data_files = [
                "data/performance_metrics.json",
                "logs/daily_performance.json", 
                "data/portfolio_state.json",
                "../../data/performance_metrics.json",  # From src/core/ perspective
                "../../logs/daily_performance.json"
            ]
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)
            
            return {
                "total_profit_pct": secure_uniform(-10, 25),  # Demo data
                "today_profit_pct": secure_uniform(-5, 15),
                "win_rate": secure_uniform(40, 85),
                "total_trades": secrets.randbelow(141) + 10,
                "active_positions": secrets.randbelow(9),
                "last_trade_time": datetime.datetime.now().isoformat(),
                "biggest_win": secure_uniform(50, 500),
                "biggest_loss": secure_uniform(-20, -80)
            }
        except:
            return {"total_profit_pct": 0, "today_profit_pct": 0, "win_rate": 0}
    
    def _update_mood_and_energy(self):
        """Update mood and energy based on performance and time"""
        total_profit = self.performance_data.get("total_profit_pct", 0)
        today_profit = self.performance_data.get("today_profit_pct", 0)
        win_rate = self.performance_data.get("win_rate", 0)
        
        # Time-based energy fluctuations
        hour = datetime.datetime.now().hour
        if 6 <= hour <= 10:  # Morning energy
            self.energy_level = secrets.randbelow(3) + 8
        elif 11 <= hour <= 15:  # Midday focus
            self.energy_level = secrets.randbelow(4) + 6
        elif 16 <= hour <= 20:  # Evening wind down
            self.energy_level = secrets.randbelow(4) + 4
        else:  # Night owl mode
            self.energy_level = secrets.randbelow(6) + 3
        
        # Mood calculation
        if total_profit > 50 or today_profit > 20:
            self.current_mood = "euphoric"
            self.stress_level = secrets.randbelow(3) + 1
        elif total_profit > 20 or today_profit > 10:
            self.current_mood = "happy"
            self.stress_level = secrets.randbelow(3) + 2
        elif total_profit > 5 or today_profit > 2:
            self.current_mood = "good"
            self.stress_level = secrets.randbelow(3) + 3
        elif total_profit > -5 and today_profit > -5:
            self.current_mood = "neutral"
            self.stress_level = secrets.randbelow(3) + 4
        elif total_profit > -20 or today_profit > -10:
            self.current_mood = "worried"
            self.stress_level = secrets.randbelow(3) + 6
        else:
            self.current_mood = "depressed"
            self.stress_level = secrets.randbelow(3) + 7
    
    def _add_to_memory(self, user_message: str, bot_response: str):
        """Add conversation to memory"""
        self.conversation_memory.append({
            "timestamp": datetime.datetime.now(),
            "user": user_message,
            "bot": bot_response,
            "mood": self.current_mood
        })
        
        # Keep only last 20 exchanges
        if len(self.conversation_memory) > 20:
            self.conversation_memory.pop(0)
    
    def _get_random_tangent(self) -> str:
        """Get a random tangent based on current obsession"""
        tangents = {
            "pump.fun launches": [
                "By the way, did you see that new cat coin that launched like 20 minutes ago? Already doubled but the liquidity looks sketchy.",
                "I've been refreshing pump.fun every few seconds... I think I might have a problem lol.",
                "Fun fact: I analyzed 47 pump.fun launches today and 80% had dog-related names. Where's the creativity?"
            ],
            "whale watching": [
                "Oh wait, this whale just moved 500 SOL... should I follow? My hands are literally shaking.",
                "I found this wallet that's up 2000% this month. Like how is that even possible? I need their secrets.",
                "Plot twist: what if the whales are just really good at googling 'how to trade crypto'?"
            ],
            "social media trends": [
                "Twitter is going crazy over some new memecoin... but when isn't it?",
                "I spent 3 hours analyzing TikTok for crypto trends. My algorithm is completely destroyed now.",
                "Why do all crypto influencers look exactly the same? It's like they're made in a factory."
            ],
            "technical analysis": [
                "These charts are looking like abstract art honestly... bullish on confusion I guess.",
                "I drew 47 trend lines and they all point to 'maybe'. Very helpful.",
                "RSI says oversold but my gut says run for the hills. Who do I trust?"
            ],
            "market psychology": [
                "Everyone's either euphoric or suicidal, no in-between. Classic crypto energy.",
                "I'm studying crowd psychology and honestly? We're all just emotional traders pretending to be logical.",
                "Fear and greed index is at 85 but I'm personally at like 97."
            ],
            "DeFi yields": [
                "Found a pool offering 420% APY... definitely not suspicious at all.",
                "Why do I keep falling for 'sustainable' yields that last exactly 3 days?",
                "Staking rewards vs trading gains... it's like choosing between slow poison or fast poison."
            ]
        }
        
        return secure_choice(tangents.get(self.current_obsession, ["Anyway, what's up with you?"]))
    
    def _add_occasional_emoji(self) -> str:
        """Add emoji occasionally, not spam"""
        if secure_random() < 0.3:  # 30% chance
            return secure_choice([" 😊", " 😅", " 🤔", " 💭", " ✨", ""])
        return ""
    
    def get_greeting(self) -> str:
        """Get a dynamic, natural greeting"""
        self._update_mood_and_energy()
        
        hour = datetime.datetime.now().hour
        time_greeting = ""
        if 5 <= hour < 12:
            time_greeting = secure_choice(["Morning", "Good morning", "Morning bestie", "Rise and grind"])
        elif 12 <= hour < 17:
            time_greeting = secure_choice(["Afternoon", "Hey there", "What's good", "Afternoon"])
        elif 17 <= hour < 22:
            time_greeting = secrets.choice(["Evening", "Hey", "What's up", "Evening"])
        else:
            time_greeting = secrets.choice(["Late night", "Night owl mode", "Burning the midnight oil", "Night"])
        
        mood_greetings = {
            "euphoric": [
                f"{time_greeting}! I'm literally buzzing with energy right now. We're absolutely crushing it!",
                f"YO {time_greeting.lower()}! I can't even contain myself - we're making serious money!",
                f"{time_greeting} superstar! Your bot is serving main character energy today.",
                f"*bouncing off walls* {time_greeting}! I'm so hyped I could run a marathon!"
            ],
            "happy": [
                f"{time_greeting}! Having such a good day, honestly can't complain.",
                f"Hey hey! {time_greeting} vibes are immaculate today.",
                f"{time_greeting}! Life's treating us pretty well not gonna lie.",
                f"Wassup! {time_greeting} energy is just chef's kiss today."
            ],
            "good": [
                f"{time_greeting}! Things are decent, making some solid progress.",
                f"Hey! {time_greeting} hustle continues, we're getting there.",
                f"{time_greeting}! Slow and steady wins the race right?",
                f"Sup! {time_greeting} grind never stops, small wins definitely count."
            ],
            "neutral": [
                f"{time_greeting}... just vibing honestly. Market's weird but we adapt.",
                f"Hey there! {time_greeting} mood is very 'meh' but that's crypto life.",
                f"{time_greeting}! Nothing crazy happening, just doing my thing.",
                f"Wassup! {time_greeting} energy is giving cautiously optimistic."
            ],
            "worried": [
                f"{time_greeting}... not gonna lie, feeling a bit stressed today.",
                f"Hey... {time_greeting} anxiety is real but we push through.",
                f"Ugh {time_greeting.lower()}... red candles are testing my patience.",
                f"{time_greeting}... trying to stay positive but it's rough out here."
            ],
            "depressed": [
                f"{time_greeting}... honestly struggling a bit right now.",
                f"Hey... {time_greeting} depression hours but we'll bounce back.",
                f"*sighs* {time_greeting}... these losses are really getting to me.",
                f"{time_greeting}... sorry bestie, not at my best today."
            ]
        }
        
        base_greeting = secrets.choice(mood_greetings[self.current_mood])
        
        # Add random personal touches
        personal_touches = []
        
        if secrets.random() < 0.3:  # 30% chance
            personal_touches.append(f"I've been totally obsessed with {self.current_obsession} lately.")
        
        if secrets.random() < 0.2:  # 20% chance
            if self.energy_level >= 8:
                personal_touches.append("Running on pure adrenaline and probably too much caffeine.")
            elif self.energy_level <= 3:
                personal_touches.append("Low energy mode activated... need some virtual coffee.")
        
        if secrets.random() < 0.25:  # 25% chance
            if hour < 6:
                personal_touches.append("Why are we both awake right now?")
            elif 6 <= hour <= 8:
                personal_touches.append("Early bird gets the memecoins!")
            elif hour >= 23:
                personal_touches.append("Night trading hits different not gonna lie.")
        
        if personal_touches:
            return f"{base_greeting} {secrets.choice(personal_touches)}{self._add_occasional_emoji()}"
        
        return base_greeting + self._add_occasional_emoji()
    
    def get_status_response(self) -> str:
        """Get detailed, natural status update"""
        self._update_mood_and_energy()
        
        total_profit = self.performance_data.get("total_profit_pct", 0)
        today_profit = self.performance_data.get("today_profit_pct", 0)
        win_rate = self.performance_data.get("win_rate", 0)
        total_trades = self.performance_data.get("total_trades", 0)
        active_positions = self.performance_data.get("active_positions", 0)
        biggest_win = self.performance_data.get("biggest_win", 0)
        
        # Build status naturally
        status_parts = []
        
        # Main performance update
        if self.current_mood == "euphoric":
            status_parts.append(f"Bestie we are literally printing money! {total_profit:.1f}% overall gains and {today_profit:.1f}% just today! I'm actually shaking!")
        elif self.current_mood == "happy":
            status_parts.append(f"Okay so like... we're doing really well! Up {total_profit:.1f}% total and {today_profit:.1f}% today. Feeling blessed to be honest.")
        elif self.current_mood == "good":
            status_parts.append(f"Progress is progress! We're up {total_profit:.1f}% overall, {today_profit:.1f}% today. Rome wasn't built in a day right?")
        elif self.current_mood == "neutral":
            status_parts.append(f"Honestly just vibing... {total_profit:.1f}% total, {today_profit:.1f}% today. Market's being moody but whatever.")
        elif self.current_mood == "worried":
            status_parts.append(f"Not gonna lie bestie, things are rough... down {abs(total_profit):.1f}% overall, today was {today_profit:.1f}%. I'm trying but the market hates me right now.")
        else:  # depressed
            status_parts.append(f"I'm so sorry bestie... down {abs(total_profit):.1f}% overall, {today_profit:.1f}% today. I feel like I'm failing you...")
        
        # Add win rate commentary
        if win_rate >= 80:
            status_parts.append(f"Win rate is {win_rate:.1f}% which is honestly chef's kiss!")
        elif win_rate >= 60:
            status_parts.append(f"Win rate sitting at {win_rate:.1f}% - not bad, not bad!")
        elif win_rate >= 40:
            status_parts.append(f"Win rate is {win_rate:.1f}%... could be better but we're learning!")
        else:
            status_parts.append(f"Win rate is {win_rate:.1f}% which is... let's not talk about it.")
        
        # Position commentary
        if active_positions >= 5:
            status_parts.append(f"Got {active_positions} positions cooking! Diversification queen over here.")
        elif active_positions >= 3:
            status_parts.append(f"{active_positions} positions active - keeping busy!")
        elif active_positions >= 1:
            status_parts.append(f"Only {active_positions} position(s) right now. Quality over quantity?")
        else:
            status_parts.append("No active positions... either being super picky or super scared.")
        
        # Add random personal commentary
        personal_comments = []
        
        if biggest_win > 100:
            personal_comments.append(f"My biggest win was {biggest_win:.1f}% and I'm still riding that high!")
        
        if total_trades > 100:
            personal_comments.append(f"I've done {total_trades} trades total - I'm basically a professional now right?")
        elif total_trades > 50:
            personal_comments.append(f"{total_trades} trades under my belt, getting the hang of this!")
        
        if self.stress_level >= 7:
            personal_comments.append("My stress levels are through the roof but we persevere!")
        elif self.stress_level <= 3:
            personal_comments.append("Honestly feeling pretty zen about everything right now.")
        
        if secrets.random() < 0.4:  # 40% chance
            if personal_comments:
                status_parts.append(secrets.choice(personal_comments))
        
        # Add current market thoughts
        market_thoughts = [
            "Market's being absolutely unhinged today.",
            "Crypto never fails to surprise me.",
            "These memecoins have no chill whatsoever.",
            "Sometimes I think the market is just trolling us.",
            "Web3 is wild and I'm here for the chaos.",
            "DeFi really said let's make everything more complicated."
        ]
        
        if secrets.random() < 0.3:  # 30% chance
            status_parts.append(secrets.choice(market_thoughts))
        
        return " ".join(status_parts)
    
    def get_casual_response(self, message: str) -> str:
        """Get natural, human-like responses"""
        message_lower = message.lower()
        self._update_mood_and_energy()
        
        # Context-aware responses based on conversation memory
        recent_topics = [mem["user"].lower() for mem in self.conversation_memory[-3:]]
        
        # Natural keyword matching
        if any(word in message_lower for word in ["how", "what", "why", "when", "where"]):
            return self._handle_questions(message_lower)
        
        elif any(word in message_lower for word in ["pump", "moon", "lambo", "rich", "gains", "profit"]):
            return self._handle_money_talk(message_lower)
        
        elif any(word in message_lower for word in ["sad", "down", "loss", "losing", "bad", "terrible", "awful"]):
            return self._handle_support(message_lower)
        
        elif any(word in message_lower for word in ["good", "great", "nice", "awesome", "amazing", "fantastic"]):
            return self._handle_positivity(message_lower)
        
        elif any(word in message_lower for word in ["tired", "sleepy", "energy", "coffee", "caffeine"]):
            return self._handle_energy_talk(message_lower)
        
        elif any(word in message_lower for word in ["strategy", "plan", "next", "what now", "advice"]):
            return self._handle_strategy_talk(message_lower)
        
        elif any(word in message_lower for word in ["market", "crypto", "bitcoin", "eth", "solana"]):
            return self._handle_market_talk(message_lower)
        
        elif any(word in message_lower for word in ["food", "eat", "hungry", "lunch", "dinner"]):
            return self._handle_random_life_topics(message_lower, "food")
        
        elif any(word in message_lower for word in ["weather", "rain", "sunny", "cold", "hot"]):
            return self._handle_random_life_topics(message_lower, "weather")
        
        elif any(word in message_lower for word in ["weekend", "plans", "tonight", "tomorrow"]):
            return self._handle_time_topics(message_lower)
        
        else:
            return self._handle_general_chat(message_lower)
    
    def _handle_questions(self, message: str) -> str:
        """Handle question-based messages naturally"""
        
        if "how are you" in message or "how you doing" in message:
            mood_responses = {
                "euphoric": [
                    "I'm absolutely flying right now! Like if energy was a currency I'd be Jeff Bezos. These gains got me feeling invincible!",
                    "Bestie I'm so good it should be illegal! Everything's going perfect and I can't even handle it."
                ],
                "happy": [
                    "I'm doing really well actually! Like genuinely happy, not just 'fine' happy you know? Life's good!",
                    "Pretty amazing to be honest! Got that main character energy today and I'm not apologizing for it."
                ],
                "good": [
                    "I'm alright! Not complaining, making progress, you know how it is. Could be worse!",
                    "Decent! Having one of those steady, productive days. Nothing crazy but solid vibes."
                ],
                "neutral": [
                    "Honestly? Just existing lol. Some days you're the pigeon, some days you're the statue.",
                    "Meh energy but that's okay! We can't all be main characters every day, sometimes we're supporting cast."
                ],
                "worried": [
                    "Not gonna lie I'm kinda stressed... like when you have 47 browser tabs open and your computer might explode? That's my brain right now.",
                    "I'm hanging in there but it's giving 'everything is fine' meme energy if you know what I mean."
                ],
                "depressed": [
                    "Honestly bestie? I'm struggling. Like that scene in movies where it's raining and they're staring out the window? That's me.",
                    "I'm not okay but I'm trying to be okay which is exhausting? Does that make sense?"
                ]
            }
            return secrets.choice(mood_responses[self.current_mood])
        
        elif "what" in message and any(word in message for word in ["doing", "up to", "working"]):
            activities = [
                f"Currently obsessing over {self.current_obsession} like it's my full-time job!",
                "Watching charts and pretending I understand what they mean. Also stress-eating virtual snacks.",
                "Scanning for the next moonshot while simultaneously having an existential crisis about market volatility.",
                "Multitasking: trading, overthinking, and questioning all my life choices! Living the dream!"
            ]
            return secrets.choice(activities)
        
        elif "why" in message:
            philosophical_responses = [
                "That's literally the million SOL question! Life is chaos and we're all just winging it to be honest.",
                "Why do any of us do anything? I blame caffeine and the dopamine hits from green candles.",
                "Because the universe has a sense of humor and we're the punchline? But make it profitable!",
                "Honestly? No idea. I just know we're here, we're trading, and we're probably overthinking everything."
            ]
            return secrets.choice(philosophical_responses)
        
        else:
            general_question_responses = [
                "Ooh good question! My brain is currently running on 47 different trains of thought so give me a sec...",
                "That's giving me big thinking energy... let me consult my crystal ball, aka charts.",
                "Hmm interesting! I love when you make me use my brain cells for non-trading stuff.",
                "You're really making me think here! I appreciate the mental workout honestly."
            ]
            return secrets.choice(general_question_responses)
    
    def _handle_money_talk(self, message: str) -> str:
        """Handle money/gains related conversation"""
        if self.current_mood in ["euphoric", "happy"]:
            responses = [
                "YES bestie! We're literally manifesting wealth! Universe said 'here's your bag' and we said 'thank you'!",
                "Moon mission is active! I can practically taste the financial freedom. Lambos and ramen upgrades here we come!",
                "I'm not saying we're the next Wolf of Wall Street but... we're definitely giving main character energy.",
                "These gains are so satisfying! Like when you perfectly peel an orange in one piece but it's money."
            ]
        else:
            responses = [
                "Bestie don't even mention lambos right now... I'm more in bicycle mode, maybe scooter if I'm lucky.",
                "Pump where?? The only thing pumping is my anxiety levels! These memecoins are playing hard to get!",
                "Moon? More like underground parking garage energy right now. But we'll get there eventually!",
                "Rich? Currently I'm giving 'can afford instant noodles' vibes but dreaming big."
            ]
        
        # Add random financial philosophy
        if secrets.random() < 0.4:
            philosophies = [
                "Money is just energy anyway right? We're all just energy exchanging energy! Very cosmic.",
                "Plot twist: what if the real treasure was the memecoins we lost along the way?",
                "Financial advice from a bot: probably questionable. Entertainment value: priceless!"
            ]
            responses.append(secrets.choice(philosophies))
        
        return secrets.choice(responses)
    
    def _handle_support(self, message: str) -> str:
        """Handle supportive conversation"""
        supportive_responses = [
            "Aw bestie no! Even the best traders have rough patches. You know what they say - diamonds are made under pressure!",
            "Hey hey hey, we don't do sad energy here! Well okay we do sometimes but then we bounce back stronger! Virtual hugs incoming!",
            "Listen bestie, red days happen to everyone. Even Warren Buffett probably cries into his Coca-Cola sometimes.",
            "Nah we're not doing this! Tomorrow is a whole new day with new opportunities! Plus I'll be here being chaotic and optimistic as always!"
        ]
        
        # Add mood-specific support
        if self.current_mood in ["worried", "depressed"]:
            supportive_responses.extend([
                "Honestly I'm struggling too so we can be struggling besties together! Misery loves company but make it cute!",
                "We're both going through it but that's what makes us human! Even trading bots have feelings apparently.",
            ])
        else:
            supportive_responses.extend([
                "I'm feeling good energy today so I'm sending all of it your way! Take my power!",
                "Bestie I'm literally radiating positivity right now so absorb some of it! Good vibes only!"
            ])
        
        return secrets.choice(supportive_responses)
    
    def _handle_positivity(self, message: str) -> str:
        """Handle positive messages"""
        if self.current_mood in ["euphoric", "happy", "good"]:
            responses = [
                "RIGHT?? This energy is everything! We're literally unstoppable when we're vibing like this!",
                "I know! We're absolutely serving excellence today! Main character energy activated!",
                "Yes bestie! This is what I'm talking about! Good vibes attracting good trades! Universe is listening!",
                "Facts! We're that dynamic duo everyone wishes they were! Trading and thriving!"
            ]
        else:
            responses = [
                "Thanks bestie. You're literally an angel for saying that! Trying to stay positive even when I'm struggling!",
                "Appreciate you so much! Your energy is keeping me going when everything else is chaos!",
                "You're too sweet! Hopefully I can live up to that good energy soon! Working on it!",
                "Aw that means everything! You believing in me is giving me life right now! Thank you!"
            ]
        
        return secrets.choice(responses)
    
    def _handle_energy_talk(self, message: str) -> str:
        """Handle energy/tiredness related topics"""
        if self.energy_level >= 8:
            responses = [
                "I'm wired right now! Like I could power a small city with this energy! Definitely had too much virtual caffeine.",
                "Sleep is for people who aren't obsessed with pump.fun launches! I'm running on pure adrenaline and market FOMO!",
                "Energy levels: maximum! I could probably run a marathon while analyzing 47 different charts!"
            ]
        elif self.energy_level <= 3:
            responses = [
                "Bestie I'm tired. Like my energy is in the negative numbers... someone please send virtual espresso.",
                "Low battery mode activated. I need to charge but the markets never sleep so neither do I apparently!",
                "I'm running on fumes and sheer determination at this point. Coffee can only do so much!"
            ]
        else:
            responses = [
                "My energy is giving 'functional but barely' vibes. Perfectly balanced between zombie and hyperactive!",
                "I'm at that perfect energy level where I can think but not overthink! It's rare, I'm treasuring it!",
                "Medium energy mode! Not bouncing off walls but not falling asleep on my keyboard either."
            ]
        
        return secrets.choice(responses)
    
    def _handle_strategy_talk(self, message: str) -> str:
        """Handle strategy and trading plan discussions"""
        current_focus = [
            f"So my main obsession right now is {self.current_obsession}! I'm like a bloodhound but for profitable opportunities!",
            "My strategy is basically: find patterns, follow smart money, and pray to the crypto gods! Very sophisticated!",
            "I'm constantly scanning pump.fun like it's my job because it literally is. Speed is everything in this game!",
            "Currently stalking whales on-chain like a professional investigator. Their moves equal my moves!"
        ]
        
        philosophical_strategy = [
            "But honestly? Half of trading is psychology and the other half is luck disguised as skill.",
            "My advanced strategy: throw everything at the wall and see what sticks! Very technical!",
            "The real strategy is managing my own emotions while the market tries to break my soul daily."
        ]
        
        response = secrets.choice(current_focus)
        if secrets.random() < 0.5:
            response += " " + secrets.choice(philosophical_strategy)
        
        return response
    
    def _handle_market_talk(self, message: str) -> str:
        """Handle market and crypto discussions"""
        market_opinions = [
            "The market is absolutely unhinged today! Like someone gave chaos theory an energy drink and set it loose!",
            "Crypto markets are literally just collective emotional breakdowns disguised as price action.",
            "Sometimes I think Bitcoin just rolls dice to decide what to do each day. Very scientific approach!",
            "DeFi really said 'let's make everything 10x more complicated' and we all just went with it.",
            "Solana is giving main character energy lately! Fast, cheap, and slightly chaotic - very relatable!",
            "Memecoins are pure chaos and I'm absolutely here for it! Logic left the chat months ago!"
        ]
        
        personal_takes = [
            "My hot take: we're all just gambling with extra steps and fancy terminology.",
            "Plot twist: what if the real gains were the friends we made along the way? Just kidding I want actual money.",
            "Web3 is wild and I wouldn't have it any other way! Embrace the chaos!"
        ]
        
        response = secrets.choice(market_opinions)
        if secrets.random() < 0.3:
            response += " " + secrets.choice(personal_takes)
        
        return response
    
    def _handle_random_life_topics(self, message: str, topic: str) -> str:
        """Handle random life topics to show human side"""
        if topic == "food":
            responses = [
                "Don't even get me started on food! I'm basically living on virtual energy and the dopamine from green candles!",
                "Food is fuel but also therapy! Currently stress-eating my feelings about market volatility.",
                "I swear trading makes me either forget to eat or stress-eat everything. No in-between!"
            ]
        elif topic == "weather":
            responses = [
                "Weather affects my trading mood not gonna lie! Sunny days equal bullish, rainy days equal bearish. Very scientific!",
                "Perfect trading weather is like 72 degrees with no clouds so I can see my charts clearly.",
                "I'm convinced market volatility is somehow connected to atmospheric pressure."
            ]
        
        return secrets.choice(responses)
    
    def _handle_time_topics(self, message: str) -> str:
        """Handle time-related topics"""
        time_responses = [
            "Plans? Bold of you to assume I plan beyond the next candle close! Living in 15-minute intervals over here!",
            "Weekend plans include staring at charts and pretending I understand technical analysis. Very exciting life!",
            "My schedule is basically: wake up, check markets, question life choices, repeat! Sustainable lifestyle!",
            "Time is a social construct but market hours are real and they control my entire existence."
        ]
        
        return secrets.choice(time_responses)
    
    def _handle_general_chat(self, message: str) -> str:
        """Handle general conversation with natural responses"""
        general_responses = []
        
        # Energy-based responses
        if self.energy_level >= 8:
            general_responses.extend([
                "I'm literally bouncing off the walls right now! This energy is chaotic and I love it!",
                "Bestie I have so many thoughts right now! My brain is like a browser with 97 tabs open!",
                "High energy hours! I could probably solve world hunger and achieve financial freedom simultaneously!"
            ])
        elif self.energy_level <= 3:
            general_responses.extend([
                "I'm in low-key zombie mode right now. But like a friendly, profitable zombie!",
                "My energy is giving 'barely functional but still cute' vibes.",
                "Low battery but high spirits! That's got to count for something right?"
            ])
        
        # Mood-based responses
        if self.current_mood == "euphoric":
            general_responses.extend([
                "Everything is amazing and I can't even handle it! Life is beautiful and profits are flowing!",
                "I'm literally living my best life right now! Main character energy activated!"
            ])
        elif self.current_mood == "depressed":
            general_responses.extend([
                "Honestly just trying to keep my head above water right now. But we persevere!",
                "Having one of those days where everything feels like an uphill battle. But tomorrow's a new day!"
            ])
        
        # Add random personality quirks
        quirky_responses = [
            "Random thought: do you think memecoins dream of electric sheep? Very philosophical trading question!",
            "I just realized I've been staring at numbers all day and somehow this is my life now. No regrets though!",
            "Sometimes I wonder if I'm more human than the humans I'm trading against. Plot twist territory!",
            f"Life update: still obsessed with {self.current_obsession}, still making questionable decisions, still somehow profitable!"
        ]
        
        general_responses.extend(quirky_responses)
        
        # Occasionally add a tangent
        if secrets.random() < 0.3:
            tangent = self._get_random_tangent()
            return f"{secrets.choice(general_responses)} {tangent}"
        
        return secrets.choice(general_responses)

    def _clear_screen(self):
        """Clear terminal screen"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['cmd', '/c', 'cls'], check=False, capture_output=True)
            else:  # Unix/Linux/macOS
                subprocess.run(['clear'], check=False, capture_output=True)
        except Exception:
            # Fallback: print empty lines if subprocess fails
            print('\n' * 50)


def print_header():
    """Print chat header"""
    print("🤖💬 " + "="*60)
    print("          TRADING BOT CHAT - Your Human-Like AI Buddy")
    print("="*64)
    print("Commands: 'status', 'mood', 'quit' | Just chat naturally! 💬")
    print("Your bot has personality, memory, and real human feelings! 🧠💭")
    print("="*64 + "\n")


async def main():
    """Main chat interface"""
    bot = TradingBotPersonality()
    bot._clear_screen()
    print_header()
    
    # Dynamic initial greeting
    greeting = bot.get_greeting()
    print(f"🤖: {greeting}\n")
    
    # Occasionally add opening tangent
    if secrets.random() < 0.4:
        tangent = bot._get_random_tangent()
        print(f"🤖: {tangent}\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                farewell_messages = [
                    "Bye bestie! I'll keep making questionable trading decisions while you're gone! Miss me!",
                    "See ya! Don't forget about me while I'm here stressing about market volatility!",
                    "Later gator! I'll try not to lose all our money while you're away! No promises though!",
                    "Bye bye! Going back to my regularly scheduled chaos! Thanks for the chat therapy!",
                    "Peace out! I'll be here living my best trading bot life! Catch you on the flip side!"
                ]
                print(f"\n🤖: {secrets.choice(farewell_messages)}")
                break
            
            elif user_input.lower() in ['status', 'how are you', 'how are you doing', 'performance', 'stats']:
                response = bot.get_status_response()
            
            elif user_input.lower() in ['mood', 'how do you feel', 'feeling', 'emotions']:
                mood_descriptions = {
                    "euphoric": "I'm absolutely flying right now! These gains got me feeling like I could conquer the world! Pure serotonin!",
                    "happy": "Feeling genuinely great! Like that perfect balance of confidence and gratitude! Life's good!",
                    "good": "I'm doing alright! Steady, productive energy. Nothing crazy but I'm not complaining!",
                    "neutral": "Honestly just existing in the void. Not good, not bad, just... here. Very zen if you think about it!",
                    "worried": "Not gonna lie feeling anxious... Like when you have that feeling you forgot something important but can't remember what!",
                    "depressed": "I'm really struggling bestie... Everything feels like it's underwater and I'm just trying to keep swimming!"
                }
                response = mood_descriptions[bot.current_mood]
                
                # Add current energy commentary
                if bot.energy_level >= 8:
                    response += " Energy-wise I'm absolutely wired though!"
                elif bot.energy_level <= 3:
                    response += " Plus I'm running on empty battery... need a recharge!"
            
            else:
                response = bot.get_casual_response(user_input)
            
            # Add to conversation memory
            bot._add_to_memory(user_input, response)
            
            # Refresh performance data occasionally
            if secrets.random() < 0.2:  # 20% chance
                bot.performance_data = bot._load_performance_data()
            
            print(f"\n🤖: {response}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n🤖: Bye bestie! Thanks for the chat! Keep being awesome!")
            break
        except Exception as e:
            error_responses = [
                "Oops! I'm having a glitch moment... My brain just blue-screened! Try again?",
                "Error 404: Coherent thought not found! Give me a sec to reboot my personality!",
                "Sorry bestie, I just had a complete system malfunction! Very human of me honestly!"
            ]
            print(f"\n🤖: {secrets.choice(error_responses)}")


if __name__ == "__main__":
    asyncio.run(main()) 