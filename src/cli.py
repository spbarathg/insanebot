import asyncio
import logging
import sys
import argparse
from src.core.chat.chat_interface import ChatInterface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the chat interface."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ant Princess Trading Bot Chat Interface")
    parser.add_argument("--test", action="store_true", help="Test mode - just initialize and exit")
    parser.add_argument("--metrics", action="store_true", help="Show available metrics commands")
    args = parser.parse_args()
    
    if args.metrics:
        print("\n📊 Available Chat Metrics Commands:")
        print("• Type 'metrics' during chat to see session stats")
        print("• Type 'exit' to quit the chat")
        print("• Chat naturally about trading and markets")
        return
    
    try:
        # Initialize chat interface
        chat = ChatInterface()
        await chat.initialize()
        
        if args.test:
            print("✅ Chat interface initialized successfully!")
            await chat.close()
            return
        
        print("\n🚀 Welcome to the Ant Princess Trading Bot! 🐜")
        print("Just chat naturally with me about trading and the market.")
        print("Type 'metrics' for stats or 'exit' to quit.\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nThanks for chatting! Take care! 👋")
                break
            elif user_input.lower() == 'metrics':
                try:
                    metrics = await chat.get_chat_metrics("user1")
                    print("\nQuick Stats:")
                    print(f"Chat Duration: {metrics.get('session_duration', 0):.1f} seconds")
                    print(f"Messages: {metrics.get('message_count', 0)}")
                    print(f"Current Mood: {metrics.get('mood', 'neutral')}")
                    print(f"Tone: {metrics.get('tone', 'friendly')}")
                    print(f"Last Market Check: {metrics.get('last_market_check', 'Never')}\n")
                except Exception as e:
                    print(f"Could not get metrics: {str(e)}\n")
                continue
                
            # Process message
            try:
                response = await chat.process_message("user1", user_input)
                if response:
                    print(f"Bot: {response}\n")
                else:
                    print("Bot: I'm thinking... 🤔\n")
            except Exception as e:
                print(f"Bot: Sorry, I had a small hiccup: {str(e)}\n")
            
    except KeyboardInterrupt:
        print("\nCatch you later! 👋")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        print(f"❌ Error: {str(e)}")
    finally:
        if 'chat' in locals():
            await chat.close()
        
if __name__ == "__main__":
    asyncio.run(main()) 