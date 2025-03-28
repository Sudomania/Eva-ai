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
        
        # Assuming you have a custom model and token-to-word mappings
    def decode_response(model_output_tensor, vocab):
        # Convert tensor to indices (e.g., [23, 5, 10])
        indices = model_output_tensor.argmax(dim=-1).tolist()[0]  
        
        # Map indices to words using your vocabulary
        words = [vocab[index] for index in indices if index in vocab]  
        
        # Combine into a sentence
        return " ".join(words)  

    # Usage in your Discord bot
    async def on_message(message):
        if message.author == client.user:
            return

        # Get model output (tensor)
        input_tensor = your_custom_tokenizer(message.content)  
        output_tensor = your_model.generate(input_tensor)  

        # Decode to text
        response = decode_response(output_tensor, your_vocab_dict)  

        # Fallback if empty
        if not response.strip():
            response = "I didn't understand that."  

        await message.channel.send(response)


client = MyBot(intents=intents)
client.run(DISCORD_TOKEN)
