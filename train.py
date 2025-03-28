import torch
import torch.optim as optim
import torch.nn as nn
from model import TransformerChatbot  # Import your model
from tokenizer import SimpleTokenizer  # Import your tokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Define the training loop
def train_model(model, tokenizer, data_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Move the model to the correct device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in data_loader:
            # Move data to the same device as the model
            src = torch.tensor([src]).to(device)  # Convert list to tensor
            tgt = torch.tensor([tgt]).to(device)  # Convert list to tensor

            optimizer.zero_grad()

            # Forward pass through the model (model input/output)
            output = model(src, tgt[:-1])  # Exclude the last token for prediction

            # Compute the loss
            loss = criterion(output.view(-1, model.vocab_size), tgt[1:].view(-1))  # Shift for the target
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}")

# Initialize the tokenizer
tokenizer = SimpleTokenizer()  # Initialize your tokenizer object

# Get vocab size from tokenizer
vocab_size = len(tokenizer.vocab)  # Ensure your tokenizer has the vocab attribute
embed_size = 256                   # Define embedding size
num_heads = 8                      # Define number of attention heads
num_layers = 4                     # Define number of transformer layers
hidden_size = 512                  # Define hidden size

# Initialize the model with the required parameters
model = TransformerChatbot(vocab_size, embed_size, num_heads, num_layers, hidden_size)

# Load your dataset and prepare the data loader
# (Make sure to preprocess your dataset correctly for tokenization)
dataset = load_dataset("neifuisan/Neuro-sama-QnA")  # For example, your dataset
train_data, val_data = prepare_data(dataset, tokenizer)  # This function should be implemented

# Create DataLoader
data_loader = DataLoader(list(zip(train_data, val_data)), batch_size=32, shuffle=True)

# Train your model
train_model(model, tokenizer, data_loader)
