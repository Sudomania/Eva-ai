import discord
from model import TransformerChatbot
from tokenizer import SimpleTokenizer
import torch
import logging
import sys
from dotenv import load_dotenv
import os
from datasets import load_dataset
from tokenizer import SimpleTokenizer

# Load the dataset
dataset = load_dataset("neifuisan/Neuro-sama-QnA")  # Or whatever dataset you're using

# Initialize the tokenizer with the dataset
tokenizer = SimpleTokenizer(dataset)

# Debugging absolute path for .env file
load_dotenv(dotenv_path="C:/Users/boogi/Documents/Personal Projects/Eva-ai/.env")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not DISCORD_TOKEN:
    print("DISCORD_TOKEN is not found! Check your .env file.")
    raise ValueError("DISCORD_TOKEN not found! Check your .env file.")
else:
    print(f"DISCORD_TOKEN loaded: {DISCORD_TOKEN[:5]}... (truncated for security)")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = SimpleTokenizer(dataset)  # Ensure you pass the dataset to the tokenizer

# Set up the intents
intents = discord.Intents.default()
intents.message_content = True

# Define model parameters
vocab_size = 10000
embed_size = 256
num_heads = 8
num_layers = 4
hidden_size = 512

# Initialize the model
model = TransformerChatbot(vocab_size, embed_size, num_heads, num_layers, hidden_size)

def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(message.encode('utf-8') + b'\n')

class MyBot(discord.Client):
    async def on_ready(self):
        try:
            safe_print(f'Logged in as {self.user}')
        except UnicodeEncodeError:
            safe_print(f'Logged in as {self.user.name} (UnicodeError)')
        
    async def on_message(self, message):
        if message.author == self.user:
            return

        input_text = message.content
        input_tokens = tokenizer.encode(input_text)

        # Ensure the model is in evaluation mode
        model.eval()
        
        # Convert input tokens to a tensor and add batch dimension
        input_tensor = torch.tensor(input_tokens).unsqueeze(0)

        with torch.no_grad():
            response_tensor = model(input_tensor, input_tensor)
        
        # Log the raw model output tensor
        logger.info(f"Raw model response tensor: {response_tensor}")

        # Get the most probable tokens at each position
        response_ids = torch.argmax(response_tensor, dim=-1)

        # Decode the response tokens into text
        response = tokenizer.decode(response_ids[0].tolist())

        # Log the decoded response
        logger.info(f"Decoded response: {response}")

        if len(response) > 4000:
            response = response[:4000]

        if not response.strip():
            response = "I'm sorry, I couldn't generate a response. Please try again later."

        # Sanitize the response (remove any non-printable characters)
        response = ''.join(c for c in response if 32 <= ord(c) <= 126)

        logger.info(f"Final generated response: {response}")

        await message.channel.send(response)


client = MyBot(intents=intents)
client.run(DISCORD_TOKEN)
