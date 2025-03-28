import torch
import torch.nn as nn

class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(TransformerChatbot, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=embed_size, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size
        )
        
        # Final output layer to predict the next token
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        # Embed input tokens
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # Pass through the transformer
        output = self.transformer(src, tgt)

        # Output layer to predict next token
        output = self.fc_out(output)
        return output

# Example: Initializing the model with sample hyperparameters
vocab_size = 10000  # Example vocab size, adjust as needed
embed_size = 256    # Embedding dimension
num_heads = 8       # Number of attention heads
num_layers = 4      # Number of Transformer layers
hidden_size = 512   # Hidden layer size

model = TransformerChatbot(vocab_size, embed_size, num_heads, num_layers, hidden_size)
