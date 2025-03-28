import discord
from model import TransformerChatbot  # Updated for debugging
from tokenizer import SimpleTokenizer
import torch
import logging
import sys
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")  # Ensure it's loading the correct file
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN not found! Check your .env file.")
else:
    print(f"DISCORD_TOKEN loaded: {DISCORD_TOKEN[:5]}... (truncated for security)")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = SimpleTokenizer()  # Initialize the tokenizer

# Define a simple transformer model with random weights
class SimpleTransformerChatbot(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(SimpleTransformerChatbot, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.transformer = torch.nn.Transformer(embed_size, num_heads, num_layers)
        self.fc = torch.nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        return self.fc(output)

# Set parameters
vocab_size = 30000
embed_size = 512
num_heads = 8
num_layers = 6
hidden_size = 2048

# Initialize the model
model = SimpleTransformerChatbot(vocab_size, embed_size, num_heads, num_layers, hidden_size)

# Custom print function to handle UnicodeEncodeError
def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        # Fall back to handling UnicodeEncodeError
        sys.stdout.buffer.write(message.encode('utf-8') + b'\n')

# Set up the intents
intents = discord.Intents.default()  # You can modify this based on the required events
intents.message_content = True  # This intent is required to read the message content

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
        print(f"Input tokens: {input_tokens}")  # Debugging
        input_tensor = torch.tensor(input_tokens).unsqueeze(0)  # Add batch dimension

        # Generate a response
        with torch.no_grad():
            response_tensor = model(input_tensor, input_tensor)

        if torch.isnan(response_tensor).any() or torch.isinf(response_tensor).any():
            print("Model output contains NaNs or Infs.")
            response = "Error in generating response."
        else:
            response = tokenizer.decode(response_tensor[0].tolist())
            print(f"Decoded response: {response}")  # Debugging

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
