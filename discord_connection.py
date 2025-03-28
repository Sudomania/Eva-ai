import discord
from model import TransformerChatbot
from tokenizer import SimpleTokenizer
import torch
import logging
from dotenv import load_dotenv
import os
from datasets import load_dataset
import sys

# Configure system for Unicode support
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Disable voice warnings
discord.voice_client.VoiceClient.warn_nacl = False

# Load environment
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

class MyBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize tokenizer with dataset
        self.dataset = load_dataset("neifuisan/Neuro-sama-QnA")
        self.tokenizer = SimpleTokenizer(self.dataset)
        
        # Initialize model
        self.model = TransformerChatbot(
            vocab_size=10000,
            embed_size=256,
            num_heads=8,
            num_layers=4,
            hidden_size=512
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.max_length = 50
        self.sos_token = 0
        self.eos_token = 1

    async def on_ready(self):
        try:
            user = str(self.user).encode('utf-8', errors='replace').decode('utf-8')
            print(f'Logged in as {user}')
        except Exception as e:
            print(f"Login notification failed: {str(e)}")

    def predict_response(self, text):
        """Final fixed version with correct tensor handling"""
        try:
            # 1. Prepare source tensor [1, seq_len]
            src = torch.tensor([self.tokenizer.encode(text)], 
                             dtype=torch.long).to(self.device)
            
            # 2. Initialize target with SOS token [1, 1]
            tgt = torch.tensor([[self.sos_token]], 
                             dtype=torch.long).to(self.device)
            
            # 3. Generation loop
            for _ in range(self.max_length):
                # Ensure proper shapes for transformer [seq_len, batch_size]
                src_transposed = src.transpose(0, 1)  # [seq_len, 1]
                tgt_transposed = tgt.transpose(0, 1)  # [seq_len, 1]
                
                with torch.no_grad():
                    output = self.model(src_transposed, tgt_transposed)
                
                # Get next token and reshape properly [1, 1]
                next_token = output.argmax(dim=-1)[-1].view(1, 1)
                
                # Stop if EOS token
                if next_token.item() == self.eos_token:
                    break
                
                # Append to sequence [1, seq_len+1]
                tgt = torch.cat([tgt, next_token], dim=1)
            
            # Convert to text
            indices = tgt[0, 1:].tolist()  # Skip SOS token
            words = []
            for idx in indices:
                if idx == self.eos_token:
                    break
                if hasattr(self.tokenizer, 'idx_to_word'):
                    word = self.tokenizer.idx_to_word.get(idx, f"[UNK:{idx}]")
                else:
                    word = str(idx)  # Fallback
                words.append(word)
            return " ".join(words).strip() or "I didn't understand that."

        except Exception as e:
            logging.error(f"Generation error: {str(e)}", exc_info=True)
            return "Sorry, I'm having technical difficulties."

    async def on_message(self, message):
        if message.author == self.user:
            return

        try:
            response = self.predict_response(message.content)
            await message.channel.send(response[:2000])
        except Exception as e:
            logging.error(f"Message error: {str(e)}")
            await message.channel.send("Error processing your message")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bot.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    intents = discord.Intents.default()
    intents.message_content = True
    bot = MyBot(intents=intents)
    bot.run(DISCORD_TOKEN)