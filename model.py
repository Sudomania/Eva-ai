import torch
import torch.nn as nn

class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(TransformerChatbot, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Note: PyTorch Transformer expects [seq_len, batch_size, features]
        self.transformer = nn.Transformer(
            d_model=embed_size, 
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size
        )
        
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        # Input shapes should be [seq_len, batch_size]
        src = self.embedding(src)  # [seq_len, batch_size, embed_size]
        tgt = self.embedding(tgt)  # [seq_len, batch_size, embed_size]
        
        # Transformer expects [seq_len, batch_size, embed_size]
        output = self.transformer(src, tgt)
        
        # Output shape: [seq_len, batch_size, vocab_size]
        return self.fc_out(output)