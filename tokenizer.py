class SimpleTokenizer:
    def __init__(self, pad_token="<PAD>", sos_token="<SOS>", eos_token="<EOS>", unk_token="<UNK>"):
        self.special_tokens = {
            'pad_token': pad_token,
            'sos_token': sos_token,
            'eos_token': eos_token,
            'unk_token': unk_token
        }
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def build_vocab(self, dataset):
        # Build vocabulary from dataset
        vocab = set()
        for example in dataset:
            vocab.update(example['input'].split())
            vocab.update(example['output'].split())
        
        # Add special tokens first
        for i, (key, token) in enumerate(self.special_tokens.items()):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        # Add other words
        for i, word in enumerate(vocab, start=len(self.special_tokens)):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        self.vocab_size = len(self.word_to_idx)
    
    def encode(self, text):
        return [
            self.word_to_idx.get(word, self.word_to_idx[self.special_tokens['unk_token']])
            for word in text.split()
        ]
    
    @property
    def pad_token_id(self):
        return self.word_to_idx[self.special_tokens['pad_token']]