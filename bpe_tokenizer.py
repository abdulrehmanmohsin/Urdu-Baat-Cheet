import re
import json
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer for Urdu text
    Builds vocabulary by iteratively merging most frequent character pairs
    """
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> index
        self.merges = []  # list of (pair_a, pair_b) merges in order
        self.token_to_id = {}
        self.id_to_token = {}
        
    def get_stats(self, words: Dict[str, int]) -> Counter:
        """
        Count frequency of adjacent pairs in the corpus
        
        Args:
            words: Dictionary of word -> frequency
            
        Returns:
            Counter of pairs -> frequency
        """
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_pair(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """
        Merge all occurrences of the most frequent pair
        
        Args:
            pair: The pair to merge (token_a, token_b)
            words: Dictionary of word -> frequency
            
        Returns:
            Updated words dictionary
        """
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]
        
        return new_words
    
    def train(self, corpus: List[str], verbose: bool = True):
        """
        Train BPE tokenizer on corpus
        
        Args:
            corpus: List of sentences/texts to train on
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Training BPE tokenizer with target vocab size: {self.vocab_size}")
            print(f"Corpus size: {len(corpus)} sentences")
        
        # Prepare initial vocabulary (character-level with end-of-word marker)
        word_freqs = Counter()
        for text in corpus:
            # Split into words and count frequencies
            words = text.split()
            word_freqs.update(words)
        
        # Add space between characters and </w> at the beginning (RTL - end is on left)
        words = {}
        for word, freq in word_freqs.items():
            words['</w> ' + ' '.join(list(word))] = freq
        
        # Get initial character vocabulary
        vocab = set()
        for word in words.keys():
            vocab.update(word.split())
        
        if verbose:
            print(f"Initial vocabulary size (characters): {len(vocab)}")
            print(f"Initial vocabulary: {vocab}")
            print(f"Unique words in corpus: {len(words)}")
        
        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            pairs = self.get_stats(words)
            
            if not pairs:
                if verbose:
                    print(f"No more pairs to merge at iteration {i}")
                break
            
            best_pair = max(pairs, key=pairs.get)
            words = self.merge_pair(best_pair, words)
            self.merges.append(best_pair)
            
            if verbose and (i + 1) % 500 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair[0]} + {best_pair[1]} -> {''.join(best_pair)} (freq: {pairs[best_pair]})")
        
        # Build final vocabulary
        final_vocab = set()
        for word in words.keys():
            final_vocab.update(word.split())
        
        self.vocab = sorted(list(final_vocab))
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Number of merges performed: {len(self.merges)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using learned BPE merges
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        words = text.split()
        tokens = []
        
        for word in words:
            # Start with character-level representation (RTL - end marker at beginning)
            word_tokens = ['</w>'] + list(word)
            
            # Apply merges in order
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        # Merge this pair
                        word_tokens = word_tokens[:i] + [''.join(pair)] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id.get('<unk>', 0)) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = [self.id_to_token.get(idx, '<unk>') for idx in token_ids]
        text = ''.join(tokens).replace('</w> ', ' ').replace('</w>', ' ')
        return text.strip()
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': self.merges,
            'token_to_id': self.token_to_id
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(idx): token for idx, token in enumerate(self.vocab)}
        print(f"Tokenizer loaded from {filepath}")
    
    def get_vocab_stats(self):
        """Print vocabulary statistics"""
        print(f"\n{'='*60}")
        print(f"BPE Tokenizer Statistics")
        print(f"{'='*60}")
        print(f"Vocabulary Size: {len(self.vocab)}")
        print(f"Number of Merges: {len(self.merges)}")
        print(f"\nFirst 20 tokens in vocabulary:")
        for i, token in enumerate(self.vocab[:20]):
            print(f"  {i}: '{token}'")
        print(f"\nLast 20 merge operations:")
        for i, (a, b) in enumerate(self.merges[-20:], start=len(self.merges)-20):
            print(f"  {i}: '{a}' + '{b}' -> '{a}{b}'")


def main():
    # Load the cleaned Urdu text
    print("Loading cleaned Urdu text...")
    with open('cleaned_urdu_text.txt', 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(corpus)} sentences")
    
    # Create and train tokenizer
    # You can adjust vocab_size as needed (common values: 5000, 8000, 10000, 16000, 32000)
    vocab_size = 800
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus, verbose=True)
    
    # Display statistics
    tokenizer.get_vocab_stats()
    
    # Test the tokenizer
    print(f"\n{'='*60}")
    print("Testing Tokenizer")
    print(f"{'='*60}")
    
    test_sentences = corpus[:5]
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"Original: {sentence}")
        tokens = tokenizer.tokenize(sentence)
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        
        # Test encoding/decoding
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        print(f"Encoded IDs: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"Decoded: {decoded}")
        print(f"Match: {decoded == sentence}")
    
    # Save the tokenizer
    tokenizer.save('urdu_bpe_tokenizer.json')
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Tokenizer saved to: urdu_bpe_tokenizer.json")
    print(f"You can load it later using: tokenizer.load('urdu_bpe_tokenizer.json')")


if __name__ == "__main__":
    main()

