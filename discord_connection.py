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
        
        try:
            # Load datasets
            neuro_sama = load_dataset("neifuisan/Neuro-sama-QnA")
            emotions = load_dataset("sychonix/emotion")
            
            # Initialize tokenizer
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.build_vocab(neuro_sama, emotions)
            
            # Initialize model
            self.model = TransformerChatbot(
                vocab_size=self.tokenizer.vocab_size,
                embed_size=256,
                num_heads=8,
                num_layers=4,
                hidden_size=512
            )
            
            # Set special tokens
            self.sos_token = self.tokenizer.word_to_idx['<SOS>']
            self.eos_token = self.tokenizer.word_to_idx['<EOS>']
            
            # Device setup
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.max_length = 50
            
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            raise

    async def on_ready(self):
        try:
            user = str(self.user).encode('utf-8', errors='replace').decode('utf-8')
            print(f'Logged in as {user}')
        except Exception as e:
            print(f"Login notification failed: {str(e)}")

    def predict_response(self, text):
        """Fixed version with proper token handling"""
        try:
            # Ensure text is not empty
            if not text.strip():
                return "Please say something to me!"
            
            # Tokenize input
            token_ids = self.tokenizer.encode(text)
            if not token_ids:
                return "I didn't understand that input."
            
            # Create tensors with proper shapes
            src = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            tgt = torch.tensor([[self.sos_token]], dtype=torch.long).to(self.device)
            
            # Generation loop
            for _ in range(self.max_length):
                # Prepare inputs for transformer
                src_transposed = src.transpose(0, 1)
                tgt_transposed = tgt.transpose(0, 1)
                
                with torch.no_grad():
                    output = self.model(src_transposed, tgt_transposed)
                
                # Get next token
                next_token = output.argmax(dim=-1)[-1].unsqueeze(0).unsqueeze(0)
                
                # Stop if EOS token or max length reached
                if next_token.item() == self.eos_token:
                    break
                    
                # Append to sequence
                tgt = torch.cat([tgt, next_token], dim=1)
            
            # Convert to text
            indices = tgt[0, 1:].tolist()  # Skip SOS token
            words = []
            for idx in indices:
                if idx == self.eos_token:
                    break
                word = self.tokenizer.idx_to_word.get(idx, '<UNK>')
                words.append(word)
            
            response = ' '.join(words).strip()
            return response if response else "I'm not sure how to respond to that."

        except Exception as e:
            logging.error(f"Generation error: {str(e)}", exc_info=True)
            return "Sorry, I'm having trouble responding right now."

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