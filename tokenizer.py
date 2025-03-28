class SimpleTokenizer:
    def __init__(self):
        pass  # No predefined vocabulary

    def encode(self, text):
        return [ord(c) for c in text]  # Convert characters to Unicode numbers

    def decode(self, token_ids):
        # Flatten list in case it's nested
        if any(isinstance(i, list) for i in token_ids):
            token_ids = [item for sublist in token_ids for item in sublist]

        # Convert float tokens to integers before calling chr()
        token_ids = [int(round(tid)) for tid in token_ids]

        # Filter out non-printable characters
        filtered_tokens = [tid for tid in token_ids if 32 <= tid <= 126]  # Only keep printable ASCII characters

        return ''.join(chr(tid) for tid in filtered_tokens)  # Convert back to text
