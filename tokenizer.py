class SimpleTokenizer:
    def __init__(self):
        # Initialize with default special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
        self.word_to_idx = {**self.special_tokens}
        self.idx_to_word = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)
    
    def build_vocab(self, neuro_sama, emotions):
        """Build vocabulary from both datasets"""
        vocab = set()
        
        # Process Neuro-sama dataset
        for example in neuro_sama['train']:
            if 'input' in example and 'output' in example:
                text = f"{example['input']} {example['output']}"
                vocab.update(text.split())
        
        # Process emotions dataset
        for example in emotions['train']:
            if 'text' in example and 'label' in example:
                text = f"How are you feeling about {example['text']}? I feel {example['label']} about it."
                vocab.update(text.split())
        
        # Add words to vocabulary
        for i, word in enumerate(vocab, start=self.vocab_size):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        self.vocab_size = len(self.word_to_idx)
    
    def encode(self, text):
        return [
            self.word_to_idx.get(word, self.special_tokens['<UNK>'])
            for word in text.split()
        ]
    
    def decode(self, token_ids):
        return ' '.join([self.idx_to_word.get(tid, '<UNK>') for tid in token_ids])