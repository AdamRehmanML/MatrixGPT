# Lets build a GPT tokenizer for our model from the same dataset
import re
from collections import defaultdict, Counter
import json

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}
        self.pattern = None
        
    def train(self, text):
        # Initial vocabulary is all characters/bytes
        word_counts = Counter(text.split())
        vocab = set(char for word in word_counts.keys() for char in word)
        vocab = list(vocab) + ['</w>']  # Add end of word token
        
        # Initialize token dictionaries
        for i, token in enumerate(vocab):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        def get_stats(vocab):
            pairs = defaultdict(int)
            for word, freq in word_counts.items():
                symbols = list(word) + ['</w>']
                for i in range(len(symbols)-1):
                    pairs[symbols[i], symbols[i+1]] += freq
            return pairs
        
        def merge_vocab(pair, v_in):
            v_out = {}
            bigram = re.escape(''.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            for word in v_in:
                w_out = p.sub(''.join(pair), word)
                v_out[w_out] = v_in[word]
            return v_out
        
        num_merges = self.vocab_size - len(vocab)
        for i in range(num_merges):
            pairs = get_stats(word_counts)
            if not pairs:
                break
                
            best = max(pairs, key=pairs.get)
            word_counts = merge_vocab(best, word_counts)
            
            # Add the merged token to vocabulary
            new_token = ''.join(best)
            self.token_to_id[new_token] = len(self.token_to_id)
            self.id_to_token[len(self.id_to_token)] = new_token
            self.merges[best] = new_token
            
        # Create regex pattern for tokenization
        self.pattern = re.compile("|".join(map(re.escape, self.token_to_id.keys())))
    
    def encode(self, text):
        if not self.pattern:
            raise ValueError("Tokenizer must be trained first!")
            
        tokens = []
        for word in text.split():
            word = word + '</w>'
            matches = self.pattern.finditer(word)
            word_tokens = [self.token_to_id[match.group(0)] for match in matches]
            tokens.extend(word_tokens)
        return tokens
    
    def decode(self, token_ids):
        text = ''.join(self.id_to_token[id] for id in token_ids)
        text = text.replace('</w>', ' ').strip()
        return text
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'merges': self.merges
            }, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.token_to_id = data['token_to_id']
        self.merges = data['merges']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pattern = re.compile("|".join(map(re.escape, self.token_to_id.keys())))
    def get_compression_stats(self, text):
        """Calculate compression statistics for the given text."""
        # Original text stats
        orig_chars = len(text)
        orig_bytes = len(text.encode('utf-8'))
        orig_words = len(text.split())
        
        # Tokenized stats
        tokens = self.encode(text)
        # Each token ID requires a certain number of bytes to store
        tokens_bytes = len(tokens) * 2  # Assuming 2 bytes per token ID (can handle vocab up to 65536)
        
        stats = {
            'original_characters': orig_chars,
            'original_bytes': orig_bytes,
            'original_words': orig_words,
            'token_count': len(tokens),
            'tokens_bytes': tokens_bytes,
            'character_compression_ratio': orig_chars / len(tokens),
            'byte_compression_ratio': orig_bytes / tokens_bytes,
            'tokens_per_word': len(tokens) / orig_words,
            'vocabulary_size': len(self.token_to_id)
        }
        return stats
# Example usage
def load_and_preprocess_text(pdf_path):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    # Basic preprocessing
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip().lower()
    return text





# text = load_and_preprocess_text('The_Matrix.pdf')
# tokenizer = BPETokenizer(vocab_size=1024)
# tokenizer.train(text)
# compression_ratio = tokenizer.get_compression_stats('Sigma skibidi patrick bateman')
# print(compression_ratio)
# print(tokenizer.encode("Sigma skibidi patrick bateman"))
# print(tokenizer.decode(tokenizer.encode("Sigma skibidi patrick bateman")))
