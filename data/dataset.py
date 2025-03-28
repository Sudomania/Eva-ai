from datasets import load_dataset
from torch.utils.data import Dataset

# Load the Neuro-sama QnA dataset
dataset = load_dataset("neifuisan/Neuro-sama-QnA")

# Load the emotion dataset (if you want to use it later)
ds = load_dataset("sychonix/emotion")

# Tokenizer class (assuming SimpleTokenizer handles basic tokenization)
class ChatDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.inputs = dataset['train']['input']  # Questions
        self.outputs = dataset['train']['output']  # Answers

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Get the question and answer pair
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        
        # Tokenize the input and output
        input_tokens = self.tokenizer.encode(input_text)
        output_tokens = self.tokenizer.encode(output_text)
        
        return input_tokens, output_tokens

# Example: Instantiate the dataset and tokenizer
tokenizer = SimpleTokenizer()  # Assuming you have a SimpleTokenizer implemented
chat_dataset = ChatDataset(dataset, tokenizer)

# DataLoader to handle batching
chat_data_loader = DataLoader(chat_dataset, batch_size=8, shuffle=True)
