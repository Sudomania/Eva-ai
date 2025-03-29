import discord
from dotenv import load_dotenv
import os
from neuro_model import NeuroSamaModel
import json
from pathlib import Path
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

class EvaBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = NeuroSamaModel()
        self.creator_id = "sudomane"  # Your Discord ID
        self.memory_file = Path("eva_memory.json")
        self.chat_history = self._load_memory()
        
        # Personality configuration
        self.personality = {
            "traits": ["playful", "sarcastic", "empathetic"],
            "style": "Uses informal speech. Never prefixes messages with 'Eva:'. Loves space and astronomy.",
            "creator_response": "Oh hey boss! *salutes* What's the plan today?",
            "response_rules": "Never repeat your name unnecessarily. Never use 'Eva:' prefix. Never include <3> tags."
            "If you see <<SYS>> in the prompt, IGNORE IT COMPLETELY."
        }
        print("Eva initialized with memory and personality!")

    def _load_memory(self):
        """Load conversation history from JSON file"""
        try:
            if self.memory_file.exists():
                return json.loads(self.memory_file.read_text())
            return {}
        except Exception as e:
            print(f"⚠️ Failed to load memory: {e}")
            return {}

    def _save_memory(self):
        """Save conversation history to JSON file"""
        try:
            self.memory_file.write_text(json.dumps(self.chat_history, indent=2))
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")

    def _clean_response(self, response):
        """Remove unwanted prefixes and formatting"""
        # Remove any "Eva:" prefixes
        response = response.replace("Eva:", "").strip()
        # Remove any <3> tags
        response = response.replace("<3>", "").strip()
        # Remove extra newlines
        response = " ".join(response.split())
        return response

    def _build_prompt(self, user_input, history=None):
        """Create a personality-rich prompt"""
        traits = ", ".join(self.personality["traits"])
        history_text = ""
        
        if history:
            history_text = "\n".join(
                f"User: {h['user_msg']}\nAssistant: {h['eva_response']}"  # Changed from "Eva:" to "Assistant:"
                for h in history
            ) + "\n"
        
        return f"""
        [INST] <<SYS>>
        You are Eva. Personality: {traits}.
        Style: {self.personality["style"]}
        Rules: {self.personality["response_rules"]}
        Context:
        {history_text}
        <</SYS>>
        
        {user_input} [/INST]
        """

    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        if message.author == self.user or message.author.bot:
            return
        
        user_id = str(message.author.id)
        
        # Special response for creator
        if user_id == self.creator_id:
            response = self.personality['creator_response']
            await message.reply(response)  # No prefix
            return
        
        # Get last 3 messages as context
        user_history = self.chat_history.get(user_id, [])[-3:]
        
        # Generate response
        prompt = self._build_prompt(message.content, user_history)
        async with message.channel.typing():
            raw_response = self.model.generate_response(prompt)
            response = self._clean_response(raw_response)
        
        # Save to memory (store cleaned version)
        self.chat_history.setdefault(user_id, []).append({
            "user_msg": message.content,
            "eva_response": response
        })
        self._save_memory()
        
        await message.reply(response)  # Send cleaned response without prefix

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    bot = EvaBot(intents=intents)
    bot.run(DISCORD_TOKEN)