import asyncio
import logging
import sys
from core.chat.chat_interface import ChatInterface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the chat interface."""
    try:
        # Initialize chat interface
        chat = ChatInterface()
        await chat.initialize()
        
        print("\n🚀 Welcome to the Ant Princess Trading Bot! 🐜")
        print("Just chat naturally with me about trading and the market.\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nThanks for chatting! Take care! 👋")
                break
            elif user_input.lower() == 'metrics':
                metrics = await chat.get_chat_metrics("user1")
                print("\nQuick Stats:")
                print(f"Chat Duration: {metrics['session_duration']:.1f} seconds")
                print(f"Messages: {metrics['message_count']}")
                print(f"Current Mood: {metrics['mood']}")
                print(f"Tone: {metrics['tone']}")
                print(f"Last Market Check: {metrics['last_market_check']}\n")
                continue
                
            # Process message
            response = await chat.process_message("user1", user_input)
            if response:
                print(f"Bot: {response}\n")
            
    except KeyboardInterrupt:
        print("\nCatch you later! 👋")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        await chat.close()
        
if __name__ == "__main__":
    asyncio.run(main()) 