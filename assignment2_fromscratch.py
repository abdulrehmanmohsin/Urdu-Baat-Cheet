

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import re
import json
import random
import os
import csv
import math
import time
import sys
from collections import Counter
from typing import List, Dict, Tuple
from sacrebleu.metrics import BLEU, CHRF
from tqdm.notebook import tqdm  # Use notebook version

# Set random seeds
torch.manual_seed(42)
random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

"""# Data Cleaning"""

def normalize_text(text):

    #Arabic Diacritical marks
    diacritics = [
        '\u0610', '\u0611', '\u0612', '\u0613', '\u0614', '\u0615', '\u0616',
        '\u0617', '\u0618', '\u0619', '\u061A', '\u064B', '\u064C', '\u064D',
        '\u064E', '\u064F', '\u0650', '\u0651', '\u0652', '\u0653', '\u0654',
        '\u0655', '\u0656', '\u0657', '\u0658', '\u0659', '\u065A', '\u065B',
        '\u065C', '\u065D', '\u065E', '\u065F', '\u0670', '\u06D6', '\u06D7',
        '\u06D8', '\u06D9', '\u06DA', '\u06DB', '\u06DC', '\u06DF', '\u06E0',
        '\u06E1', '\u06E2', '\u06E3', '\u06E4', '\u06E7', '\u06E8', '\u06EA',
        '\u06EB', '\u06EC', '\u06ED'
    ]

    for diacritic in diacritics:
        text = text.replace(diacritic, '')

    # Strandardize Alef and Yeh Forms
    text = re.sub('[أإآٱ]', 'ا', text)
    text = re.sub('[يىئ]', 'ی', text)

    # Remove zero-width characters
    text = text.replace('\u200c', '').replace('\u200d', '')

    # Normalize punctuation
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2026', '...').replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('`', "'")

    return text.strip()

def process_csv_file(input_file, output_file):
    """
    Process the CSV file and extract cleaned Urdu sentences using pandas
    """
    try:
        df = pd.read_csv(input_file, encoding='utf-8')

        sentences = df['sentence'].dropna().astype(str).tolist()

        print(f"Loaded {len(sentences)} sentences from CSV.")

        # Clean sentences
        cleaned_sentences = []
        for i, sentence in enumerate(sentences, start=1):
            try:
                cleaned = normalize_text(sentence)
                if cleaned:
                    cleaned_sentences.append(cleaned)
            except Exception as e:
                print(f"Error cleaning line {i}: {e}")

        # Save cleaned sentences
        pd.Series(cleaned_sentences).to_csv(output_file, index=False, header=False, encoding='utf-8')

        print(f"✅ Successfully processed {len(cleaned_sentences)} sentences")
        print(f"💾 Cleaned text saved to: {output_file}")

        # Show examples
        print("\nFirst 5 cleaned sentences:")
        for i, s in enumerate(cleaned_sentences[:5], 1):
            print(f"{i}. {s}")

    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"⚠️ Error: {e}")


#input_file = os.path.join(base_dir, "final_main_dataset.csv")
#output_file = os.path.join(base_dir, "cleaned_urdu_text.txt")

#process_csv_file(input_file, output_file)

"""# BPE TOKENIZER"""

class BPETokenizer:
    """Byte Pair Encoding tokenizer with Right to Left support"""

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

    def get_stats(self, words: Dict[str, int]) -> Counter:
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_pair(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]
        return new_words

    def train(self, corpus: List[str], verbose: bool = True):
        """Train BPE on corpus"""
        word_freqs = Counter()
        for text in corpus:
            word_freqs.update(text.split())

        # RTL-aware: </w> at beginning
        words = {}
        for word, freq in word_freqs.items():
            words['</w> ' + ' '.join(list(word))] = freq

        vocab = set()
        for word in words.keys():
            vocab.update(word.split())

        if verbose:
            print(f"Initial vocab: {len(vocab)} characters")

        num_merges = self.vocab_size - len(vocab)
        for i in range(num_merges):
            pairs = self.get_stats(words)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            words = self.merge_pair(best_pair, words)
            self.merges.append(best_pair)

        final_vocab = set()
        for word in words.keys():
            final_vocab.update(word.split())

        self.vocab = sorted(list(final_vocab))
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        if verbose:
            print(f"Final vocab: {len(self.vocab)} tokens")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using BPE"""
        words = text.split()
        tokens = []
        for word in words:
            word_tokens = ['</w>'] + list(word)
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        word_tokens = word_tokens[:i] + [''.join(pair)] + word_tokens[i + 2:]
                    else:
                        i += 1
            tokens.extend(word_tokens)
        return tokens

"""# Data Preparation"""

class UrduChatbotDataset(Dataset):
    """Dataset for Urdu chatbot with full sequence next-token prediction"""

    def __init__(self, sentences, tokenizer, max_len=50):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'

        for token in [self.pad_token, self.start_token, self.end_token, self.unk_token]:
            if token not in self.tokenizer.token_to_id:
                idx = len(self.tokenizer.token_to_id)
                self.tokenizer.token_to_id[token] = idx
                self.tokenizer.id_to_token[idx] = token

        self.pad_idx = self.tokenizer.token_to_id[self.pad_token]
        self.start_idx = self.tokenizer.token_to_id[self.start_token]
        self.end_idx = self.tokenizer.token_to_id[self.end_token]
        self.unk_idx = self.tokenizer.token_to_id[self.unk_token]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = self.tokenizer.tokenize(sentence)
        token_ids = [self.tokenizer.token_to_id.get(t, self.unk_idx) for t in tokens]

        # Split into prefix and continuation
        if len(token_ids) > 4:
            split_point = len(token_ids) // 2
            src_ids = token_ids[:split_point]
            tgt_ids = token_ids[split_point:]
        else:
            src_ids = token_ids[:-1]
            tgt_ids = token_ids[-1:]

        # --- TRUNCATE before padding ---
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len - 2]  # leave room for <START> and <END>

        # Encoder sequence
        encoder_ids = src_ids + [self.pad_idx] * (self.max_len - len(src_ids))

        # Decoder sequences
        decoder_input_ids = [self.start_idx] + tgt_ids
        decoder_target_ids = tgt_ids + [self.end_idx]

        # Pad to fixed length
        decoder_input_ids = decoder_input_ids + [self.pad_idx] * (self.max_len - len(decoder_input_ids))
        decoder_target_ids = decoder_target_ids + [self.pad_idx] * (self.max_len - len(decoder_target_ids))

        # Final safety truncation
        decoder_input_ids = decoder_input_ids[:self.max_len]
        decoder_target_ids = decoder_target_ids[:self.max_len]

        return {
            'encoder_input': torch.tensor(encoder_ids, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target_ids, dtype=torch.long)
        }

"""# Model Architecture"""

class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) #Assigns sine values to even positions in the positional encoding.
        pe[:, 1::2] = torch.cos(position * div_term) #Assigns cosine values to odd positions in the positional encoding.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] #Adding Positional Value
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

class Transformer(nn.Module):
    """Complete Transformer Encoder-Decoder Model"""
    def __init__(self, vocab_size, d_model=256, num_heads=2, d_ff=1024,
                 num_encoder_layers=2, num_decoder_layers=2, max_len=512,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Encoder
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.encoder_pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        encoder_output = x

        # Decoder
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.decoder_pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        output = self.output_projection(x)
        return output

"""# Training Classes"""

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)

        optimizer.zero_grad()
        output = model(encoder_input, decoder_input)
        output = output.reshape(-1, output.size(-1))
        target = decoder_target.reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            decoder_target = batch['decoder_target'].to(device)

            output = model(encoder_input, decoder_input)
            output = output.reshape(-1, output.size(-1))
            target = decoder_target.reshape(-1)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / len(val_loader)

def eval_bleu_chrf(
    model, tokenizer, dataset, sentences,
    max_gen_len, device, sample_size=1000, use_greedy=True, set_eval=True
):
    """
    Evaluate the 'first-half → second-half' objective:

    - Tokenize with tokenizer.tokenize and map via token_to_id (UNK fallback).
    - Split ids at mid-point; src = first half; ref = second half.
    - Generate using ONLY src half as encoder input.
    - Compare pred vs ref (BLEU/chrF).
    """
    import torch
    from sacrebleu.metrics import BLEU, CHRF

    prev_training = model.training
    if set_eval:
        model.eval()

    bleu = BLEU()
    chrf = CHRF()
    preds, refs = [], []

    pad_id   = getattr(dataset, "pad_idx", tokenizer.token_to_id.get("<PAD>"))
    bos_id   = getattr(dataset, "start_idx", tokenizer.token_to_id.get("<START>"))
    eos_id   = getattr(dataset, "end_idx", tokenizer.token_to_id.get("<END>"))
    unk_id   = getattr(dataset, "unk_idx", tokenizer.token_to_id.get("<UNK>"))

    def ids_to_text(ids):
        specials = {pad_id, bos_id, eos_id, unk_id}
        toks = [tokenizer.id_to_token.get(i, "<UNK>") for i in ids if i not in specials]
        return "".join(toks).replace("</w> ", " ").replace("</w>", " ").strip()

    n = min(sample_size, len(sentences))
    for i in range(n):
        s = sentences[i]

        # tokenize → ids (same style as Dataset)
        tok_list = tokenizer.tokenize(s)
        ids = [tokenizer.token_to_id.get(t, unk_id) for t in tok_list]
        if len(ids) < 4:
            continue

        split = len(ids) // 2
        src_ids = ids[:split]
        tgt_ids = ids[split:]

        src_text = ids_to_text(src_ids)
        ref_text = ids_to_text(tgt_ids)

        pred_text = generate_text(
            model, tokenizer, src_text, device,
            max_len=max_gen_len, temperature=1.0, greedy=use_greedy
        ).strip()

        preds.append(pred_text)
        refs.append(ref_text)

    if set_eval and prev_training:
        model.train()

    if not preds:
        return {"bleu_score": 0.0, "chrf_score": 0.0, "n_eval": 0}

    bleu_score = float(bleu.corpus_score(preds, [refs]).score)
    chrf_score = float(chrf.corpus_score(preds, [refs]).score)
    return {"bleu_score": bleu_score, "chrf_score": chrf_score, "n_eval": len(preds)}

"""# Text Generation"""

def generate_text(
    model, tokenizer, input_text, device,
    max_len=50, temperature=1.0, top_k=None, top_p=None,
    eos_token="<eos>", greedy=False
):
    """
    Autoregressive decoding for your Transformer that does NOT rely on tokenizer.encode/decode
    and does NOT call model.encoder()/model.decoder() (which don't exist).

    - Uses tokenizer.tokenize + token_to_id / id_to_token.
    - Uses <START>/<END>/<PAD> that were injected into tokenizer by the Dataset.
    - Builds masks using model.make_src_mask / model.make_tgt_mask.
    """


    model_was_training = model.training
    model.eval()

    # ---- Special IDs from tokenizer (injected by UrduChatbotDataset) ----
    pad_id   = tokenizer.token_to_id.get("<PAD>")
    bos_id   = tokenizer.token_to_id.get("<START>")
    eos_id   = tokenizer.token_to_id.get("<END>")
    unk_id   = tokenizer.token_to_id.get("<UNK>")

    # ---- Encode source exactly like in Dataset.__getitem__ ----
    src_tokens = tokenizer.tokenize(input_text)
    src_ids = [tokenizer.token_to_id.get(t, unk_id) for t in src_tokens]
    if len(src_ids) == 0:
        return ""

    # tensors
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = model.make_src_mask(src)  # (B,1,1,src_len)

    with torch.no_grad():
        # ----- Encoder forward (explicit, since model has no .encoder()) -----
        x = model.encoder_embedding(src) * math.sqrt(model.d_model)
        x = model.encoder_pos_encoding(x)
        for layer in model.encoder_layers:
            x = layer(x, src_mask)
        memory = x  # (B, S, d_model)

        # ----- Decoder loop -----
        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # (1,1)
        out_ids = []

        for _ in range(max_len):
            tgt_mask = model.make_tgt_mask(ys)  # (B,1,L,L)

            x = model.decoder_embedding(ys) * math.sqrt(model.d_model)
            x = model.decoder_pos_encoding(x)
            for layer in model.decoder_layers:
                x = layer(x, memory, src_mask, tgt_mask)

            logits = model.output_projection(x[:, -1])  # (1,V)

            if greedy:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
                # (optional) top-k / top-p could be added here if you want
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            out_ids.append(next_id)

            # append to sequence
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)

            if eos_id is not None and next_id == eos_id:
                break

    # ---- Detokenize: ids -> tokens -> text (strip specials, remove </w>) ----
    specials = {pad_id, bos_id, eos_id, unk_id}
    tokens = []
    for tid in out_ids:
        if tid in specials:
            continue
        tokens.append(tokenizer.id_to_token.get(tid, "<UNK>"))

    text = "".join(tokens).replace("</w> ", " ").replace("</w>", " ").strip()

    if model_was_training:
        model.train()

    return text

"""# Training"""

# Configuration
config = {
    'vocab_size': 5000,
    'd_model': 256,
    'num_heads': 2,
    'd_ff': 512,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'max_len': 50,
    'dropout': 0.1,
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-4,
}

# Load data
print("Loading data...")
#data_path = os.path.join(base_dir, "cleaned_urdu_text.txt")
with open("cleaned_urdu_text.txt", 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(sentences)} sentences")

# Train tokenizer
print("Training BPE tokenizer...")
tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
tokenizer.train(sentences[:5000], verbose=True)  # Subset for speed

# Prepare dataset
print("Preparing dataset...")
random.shuffle(sentences)
train_size = int(0.8 * len(sentences))
val_size = int(0.1 * len(sentences))

train_sentences = sentences[:train_size]
val_sentences = sentences[train_size:train_size + val_size]

train_dataset = UrduChatbotDataset(train_sentences, tokenizer, config['max_len'])
val_dataset = UrduChatbotDataset(val_sentences, tokenizer, config['max_len'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

vocab_size = len(tokenizer.token_to_id) + 4

# Create model
print("Creating model...")
model = Transformer(
    vocab_size=vocab_size,
    d_model=config['d_model'],
    num_heads=config['num_heads'],
    d_ff=config['d_ff'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    max_len=config['max_len'],
    dropout=config['dropout'],
    pad_idx=train_dataset.pad_idx
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
#Used these values to make training faster and less smooth, encouraging quicker adaptation to new gradients.

best_val_loss = float('inf')
train_losses = []
val_losses = []
best_bleu=-1.0

for epoch in range(config['num_epochs']):
    print(f"Epoch {epoch + 1}/{config['num_epochs']}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    ppl=math.exp(val_loss)
    metrics = eval_bleu_chrf(
    model, tokenizer, val_dataset, val_sentences,
    max_gen_len=config['max_len'], device=device,
    sample_size=2000
    )
    print(
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"PPL : {ppl:.4f} | BLEU: {metrics['bleu_score']:.4f} | chrf: {metrics['chrf_score']:.4f} "
        f"| n={metrics['n_eval']}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_save_path = "best_model_notebook.pt"
        torch.save(model.state_dict(), model_save_path)
        print("★ Best model saved!")

    if metrics['bleu_score'] > best_bleu:
        best_bleu = metrics['bleu_score']
        model_save_path = "best_BLEU_model.pt"
        torch.save(model.state_dict(), model_save_path)
        print("★ New best BLEU — saved best model!")
    # if val_loss < 0.1:
    #   break
    # if val_loss - train_loss > 0.9:
    #   break
print("Training complete!")

"""# Testing"""

# Load best model
#best_model=os.path.join(base_dir, "best_BLEU_model.pt")
model.load_state_dict(torch.load(best_model, map_location=device))

# Test generation
print("Testing generation:")
print("="*80)
for text in ["تاہم اس میں بہت سے نیے فیچرز", "آج موسم"]:
    out = generate_text(model, tokenizer, text,  device=device , max_len=30, greedy=True)
    print("Input:", text)
    print("Output:", out)
