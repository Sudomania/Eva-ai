import discord
from dotenv import load_dotenv
import os
from neuro_model import NeuroSamaModel

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

class EvaBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = NeuroSamaModel()
        self.prefix = "♥ Eva ♥: "
        print("Model loaded successfully")

    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        if message.author == self.user or message.author.bot:
            return
        
        async with message.channel.typing():
            response = self.model.generate_response(message.content)
            
        if response:
            await message.channel.send(f"{self.prefix}{response}")
        else:
            await message.channel.send(f"{self.prefix}*processing error*")

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    bot = EvaBot(intents=intents)
    bot.run(DISCORD_TOKEN)