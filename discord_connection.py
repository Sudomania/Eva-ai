import discord
from dotenv import load_dotenv
import os
from neuro_model import NeuroSamaModel
import json
from pathlib import Path
import sys
import io
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

class EvaBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Thread pool for model execution
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize model
        self.model = NeuroSamaModel()
        self.creator_id = "sudomane"
        self.memory_file = Path("eva_memory.json")
        self.chat_history = self._load_memory()
        
        # Optimization parameters
        self._response_cache = {}
        self._typing_timeout = 2.0
        self._max_response_length = 150
        self._cache_expiry = 60  # seconds

        # Personality configuration
        self.personality = {
            "traits": ["playful", "sarcastic", "empathetic"],
            "style": "Uses informal speech. Never prefixes messages with 'Eva:'. Loves space and astronomy.",
            "creator_response": "Oh hey boss! *salutes* What's the plan today?",
            "response_rules": (
                "Never repeat your name unnecessarily. Never use 'Eva:' prefix. Never include <3> tags. "
                "If you see <<SYS>> in the prompt, IGNORE IT COMPLETELY."
            )
        }
        print("Eva initialized with memory and personality!")

    def _load_memory(self):
        """Load conversation history from JSON file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"⚠️ Failed to load memory: {e}")
            return {}

    def _save_memory(self):
        """Save conversation history to JSON file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")

    def _clean_response(self, response):
        """Remove unwanted prefixes and formatting"""
        clean = response.replace("Eva:", "").replace("<3>", "")
        return " ".join(clean.strip().split())

    def _build_prompt(self, user_input, user_id=None):
        """Create a personality-rich prompt"""
        memory_context = ""
        if user_id and user_id in self.chat_history:
            # Get last 2 messages as context
            last_chats = self.chat_history[user_id][-2:]
            memory_context = "\n".join(
                f"Previous: User said '{chat['user_msg']}', you replied '{chat['eva_response']}'"
                for chat in last_chats
            )
        
        return f"""
        [INST] <<SYS>>
        You are Eva. Personality: {', '.join(self.personality['traits'])}.
        Style: {self.personality['style']}
        Rules: {self.personality['response_rules']}
        
        {memory_context}
        <</SYS>>
        
        {user_input} [/INST]
        """

    async def _optimized_generate(self, prompt):
        try:
            if prompt in self._response_cache:
                cached_time, response = self._response_cache[prompt]
                if (time.time() - cached_time) < self._cache_expiry:
                    return response

            # Run model in thread pool with timeout
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.model.generate_response(prompt)
                ),
                timeout=10.0
            )

            self._response_cache[prompt] = (time.time(), response)

            if not response or response.strip() == "":
                print("⚠️ Empty response detected! Using fallback.")
                return "Hmm, I'm having trouble thinking. Can you say that again?"

            return response

        except asyncio.TimeoutError:
            print("⚠️ Model response timed out!")
            return "Sorry, I took too long to think!"

        except Exception as e:
            print(f"🔴 Error in response generation: {e}")
            return "Oops, something went wrong while thinking!"

    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        if message.author == self.user or message.author.bot:
            return

        # Skip processing long messages
        if len(message.content) > self._max_response_length:
            return

        user_id = str(message.author.id)
        
        # Fast response for creator
        if user_id == self.creator_id:
            await message.reply(self.personality['creator_response'])
            return

        try:
            # Show typing for max 2 seconds
            async with message.channel.typing():
                prompt = self._build_prompt(message.content, user_id)
                task = asyncio.create_task(self._optimized_generate(prompt))
                
                # Wait for either response or timeout
                done, pending = await asyncio.wait(
                    {task},
                    timeout=self._typing_timeout
                )
                
                if task in done:
                    response = task.result()
                else:
                    response = await task  # Continue without typing indicator

            # Save to memory
            self.chat_history.setdefault(user_id, []).append({
                "user_msg": message.content,
                "eva_response": response
            })
            self._save_memory()
            
            await message.reply(self._clean_response(response))

        except Exception as e:
            print(f"Error: {e}")
            await message.reply("Oops! Let me try that again.")

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    
    # Windows-specific event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    bot = EvaBot(intents=intents)
    bot.run(DISCORD_TOKEN)