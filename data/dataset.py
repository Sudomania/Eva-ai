from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from tokenizer import SimpleTokenizer
import torch

class CombinedDataset(Dataset):
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and prepare both datasets
        self.neuro_sama = self._prepare_neuro_sama()
        self.emotions = self._prepare_emotions()
        self.combined = concatenate_datasets([self.neuro_sama, self.emotions])
        
    def _prepare_neuro_sama(self):
        dataset = load_dataset("neifuisan/Neuro-sama-QnA")
        return dataset['train'].map(lambda x: {
            'input': x['input'],
            'output': x['output'],
            'source': 'neuro_sama'
        })
    
    def _prepare_emotions(self):
        dataset = load_dataset("sychonix/emotion")
        return dataset['train'].map(lambda x: {
            'input': f"How are you feeling about {x['text']}?",
            'output': f"I feel {x['label']} about it.",
            'source': 'emotion'
        })
    
    def __len__(self):
        return len(self.combined)
    
    def __getitem__(self, idx):
        item = self.combined[idx]
        input_ids = self.tokenizer.encode(item['input'])
        output_ids = self.tokenizer.encode(item['output'])
        
        # Pad sequences to max_length
        input_ids = self._pad_sequence(input_ids)
        output_ids = self._pad_sequence(output_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'source': item['source']
        }
    
    def _pad_sequence(self, sequence):
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        return sequence + [self.tokenizer.pad_token_id] * (self.max_length - len(sequence))

# Initialize tokenizer with special tokens
tokenizer = SimpleTokenizer(
    pad_token="<PAD>",
    sos_token="<SOS>",
    eos_token="<EOS>",
    unk_token="<UNK>"
)

# Build vocabulary from both datasets
dataset = CombinedDataset(tokenizer)
tokenizer.build_vocab(dataset.combined)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda batch: {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'output_ids': torch.stack([x['output_ids'] for x in batch]),
        'sources': [x['source'] for x in batch]
    }
)