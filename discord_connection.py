import discord
from model import TransformerChatbot
from tokenizer import SimpleTokenizer
import torch
import logging
import sys
from dotenv import load_dotenv
import os

print(os.path.abspath(".env"))
load_dotenv(dotenv_path="C:/Users/boogi/Documents/Personal Projects/Eva-ai/.env")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
print(f"DISCORD_TOKEN: {DISCORD_TOKEN}")

if not DISCORD_TOKEN:
    print("DISCORD_TOKEN is not found! Check your .env file.")
    raise ValueError("DISCORD_TOKEN not found! Check your .env file.")
else:
    print(f"DISCORD_TOKEN loaded: {DISCORD_TOKEN[:5]}... (truncated for security)")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = SimpleTokenizer()  # Initialize the tokenizer

# Set up the intents
intents = discord.Intents.default()  # You can modify this based on the required events
intents.message_content = True  # This intent is required to read the message content

# Define model parameters (example values, adjust according to your setup)
vocab_size = 30000  # Adjust based on your tokenizer's vocabulary size
embed_size = 512    # Size of the embedding vectors
num_heads = 8       # Number of attention heads
num_layers = 6      # Number of transformer layers
hidden_size = 2048  # Size of the hidden layer

# Initialize the model with the required parameters
model = TransformerChatbot(vocab_size, embed_size, num_heads, num_layers, hidden_size)

# If your model requires loading pre-trained weights
# model.load_state_dict(torch.load('path_to_weights'))

# Custom print function to handle UnicodeEncodeError
def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        # Fall back to handling UnicodeEncodeError
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

        # Tokenize the message
        input_text = message.content
        input_tokens = tokenizer.encode(input_text)
        
        # Process the input with the model (make a simple response for now)
        model.eval()  # Set the model to evaluation mode
        input_tensor = torch.tensor(input_tokens).unsqueeze(0)  # Add batch dimension

        # Generate a response
        with torch.no_grad():
            response_tensor = model(input_tensor, input_tensor)
        
        response = tokenizer.decode(response_tensor[0].tolist())

        # Ensure response length does not exceed 4000 characters
        if len(response) > 4000:
            response = response[:4000]  # Truncate to 4000 characters

        # If response is empty, send a fallback message
        if not response.strip():  # Check if response is an empty string or contains only spaces
            response = "I'm sorry, I couldn't generate a response. Please try again later."

        # Ensure no non-printable characters in the response
        response = ''.join(c for c in response if 32 <= ord(c) <= 126)  # Only allow printable characters

        # Log the response for debugging
        logger.info(f"Generated response: {response}")

        await message.channel.send(response)




# Pass intents to the bot initialization
client = MyBot(intents=intents)
client.run(DISCORD_TOKEN)
