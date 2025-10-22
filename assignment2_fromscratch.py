
import math
import json
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------
# Device
# ------------------------------------
def get_device(device_override: Optional[str] = None):
    if device_override is not None:
        return torch.device(device_override)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------
# Normalization
# ------------------------------------
def normalize_urdu_text(text: str) -> str:
    """Normalize Urdu text by removing diacritics and standardizing forms"""
    if not text or not isinstance(text, str):
        return ""

    diacritics = [
        '\\u0610', '\\u0611', '\\u0612', '\\u0613', '\\u0614', '\\u0615', '\\u0616',
        '\\u0617', '\\u0618', '\\u0619', '\\u061A', '\\u064B', '\\u064C', '\\u064D',
        '\\u064E', '\\u064F', '\\u0650', '\\u0651', '\\u0652', '\\u0653', '\\u0654',
        '\\u0655', '\\u0656', '\\u0657', '\\u0658', '\\u0659', '\\u065A', '\\u065B',
        '\\u065C', '\\u065D', '\\u065E', '\\u065F', '\\u0670', '\\u06D6', '\\u06D7',
        '\\u06D8', '\\u06D9', '\\u06DA', '\\u06DB', '\\u06DC', '\\u06DF', '\\u06E0',
        '\\u06E1', '\\u06E2', '\\u06E3', '\\u06E4', '\\u06E7', '\\u06E8', '\\u06EA',
        '\\u06EB', '\\u06EC', '\\u06ED'
    ]
    for d in diacritics:
        text = text.replace(d, '')

    text = re.sub('[أإآٱ]', 'ا', text)
    text = re.sub('[يىئ]', 'ی', text)

    text = text.replace('\\u200c', '').replace('\\u200d', '')

    text = text.replace('\\u2018', "'").replace('\\u2019', "'")
    text = text.replace('\\u201C', '"').replace('\\u201D', '"')
    text = text.replace('\\u2026', '...').replace('\\u2013', '-').replace('\\u2014', '-')
    text = text.replace('`', "'")

    return text.strip()


# ------------------------------------
# BPE Tokenizer
# ------------------------------------
class BPETokenizer:
    """Byte Pair Encoding tokenizer with RTL support for Urdu + save/load"""
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab: List[str] = []
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

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

    def train(self, corpus: List[str], verbose: bool = False):
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

        num_merges = max(0, self.vocab_size - len(vocab))
        for _ in range(num_merges):
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
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")

    def tokenize(self, text: str) -> List[str]:
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

    # persistence
    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
            "token_to_id": self.token_to_id
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=int(data.get("vocab_size", 5000)))
        tok.vocab = data.get("vocab", [])
        tok.merges = [tuple(x) for x in data.get("merges", [])]
        tok.token_to_id = {k: int(v) for k, v in data.get("token_to_id", {}).items()}
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        return tok

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

        # Split sequence into halves (input/output)
        if len(token_ids) > 4:
            split_point = len(token_ids) // 2
            src_ids = token_ids[:split_point]
            tgt_ids = token_ids[split_point:]
        else:
            src_ids = token_ids[:-1]
            tgt_ids = token_ids[-1:]

        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len - 2]

        encoder_ids = src_ids + [self.pad_idx] * (self.max_len - len(src_ids))
        decoder_input_ids = [self.start_idx] + tgt_ids
        decoder_target_ids = tgt_ids + [self.end_idx]

        decoder_input_ids = decoder_input_ids + [self.pad_idx] * (self.max_len - len(decoder_input_ids))
        decoder_target_ids = decoder_target_ids + [self.pad_idx] * (self.max_len - len(decoder_target_ids))

        return {
            'encoder_input': torch.tensor(encoder_ids, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target_ids, dtype=torch.long),
        }
# ------------------------------------
# DatasetMeta to inject special tokens
# ------------------------------------
class DatasetMeta:
    """Holds special token IDs consistent with notebook Dataset behavior."""
    def __init__(self, tokenizer: BPETokenizer):
        # Special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'

        for t in [self.pad_token, self.start_token, self.end_token, self.unk_token]:
            if t not in tokenizer.token_to_id:
                idx = len(tokenizer.token_to_id)
                tokenizer.token_to_id[t] = idx
                tokenizer.id_to_token[idx] = t

        self.pad_idx = tokenizer.token_to_id[self.pad_token]
        self.start_idx = tokenizer.token_to_id[self.start_token]
        self.end_idx = tokenizer.token_to_id[self.end_token]
        self.unk_idx = tokenizer.token_to_id[self.unk_token]


# ------------------------------------
# Transformer (hyperparams aligned with notebook defaults)
# ------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
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
        batch, seq, d_model = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        out, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
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
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff))
        return x


class DecoderLayer(nn.Module):
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

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        self_attn_out = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        cross_attn = self.cross_attention(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn))
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=2, d_ff=512,
                 num_encoder_layers=2, num_decoder_layers=2, max_len=50,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_pos = PositionalEncoding(d_model, max_len, dropout)
        self.decoder_pos = PositionalEncoding(d_model, max_len, dropout)

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
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch, L = tgt.size()
        pad = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        sub = torch.tril(torch.ones((L, L), device=tgt.device)).bool()
        return pad & sub

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Encoder
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.encoder_pos(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        memory = x

        # Decoder
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.decoder_pos(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.output_projection(x)


# ------------------------------------
# Inference (generation) — greedy or temperature sampling
# ------------------------------------
@torch.no_grad()
def generate_text(model: Transformer,
                  tokenizer: BPETokenizer,
                  dataset_meta: DatasetMeta,
                  input_text: str,
                  device: Optional[torch.device] = None,
                  max_len: int = 50,
                  temperature: float = 1.0,
                  greedy: bool = True) -> str:
    """
    Autoregressive decoding using the same masking logic as training.
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode source like Dataset
    src_tokens = tokenizer.tokenize(input_text)
    unk_id = dataset_meta.unk_idx
    src_ids = [tokenizer.token_to_id.get(t, unk_id) for t in src_tokens]
    if len(src_ids) == 0:
        return ""

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = model.make_src_mask(src)

    # Run encoder once
    x = model.encoder_embedding(src) * math.sqrt(model.d_model)
    x = model.encoder_pos(x)
    for layer in model.encoder_layers:
        x = layer(x, src_mask)
    memory = x

    # Decoder loop
    ys = torch.tensor([[dataset_meta.start_idx]], dtype=torch.long, device=device)
    out_ids = []

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(ys)
        x = model.decoder_embedding(ys) * math.sqrt(model.d_model)
        x = model.decoder_pos(x)
        for layer in model.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)

        logits = model.output_projection(x[:, -1])  # (1, V)

        if greedy:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        out_ids.append(next_id)
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)

        if next_id == dataset_meta.end_idx:
            break

    specials = {dataset_meta.pad_idx, dataset_meta.start_idx, dataset_meta.end_idx, dataset_meta.unk_idx}
    toks = [tokenizer.id_to_token.get(tid, "<UNK>") for tid in out_ids if tid not in specials]
    return "".join(toks).replace("</w> ", " ").replace("</w>", " ").strip()


# ------------------------------------
# Resource Loading for Streamlit
# ------------------------------------
def _build_model_for_tokenizer(tokenizer: BPETokenizer,
                               dataset_meta: DatasetMeta,
                               config: Optional[dict] = None) -> Transformer:
    """
    Build a Transformer using vocab size derived from tokenizer AFTER special tokens
    have been injected via DatasetMeta. Defaults match notebook config.
    """
    vocab_size = len(tokenizer.token_to_id)
    cfg = dict(
        d_model=256,
        num_heads=2,
        d_ff=512,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_len=50,
        dropout=0.3
    )
    if config:
        cfg.update(config)

    model = Transformer(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
        pad_idx=dataset_meta.pad_idx
    )
    return model


def load_resources(checkpoint_path: str = "best_BLEU_model.pt",
                   tokenizer_path: Optional[str] = None,
                   device_override: Optional[str] = None,
                   config_override: Optional[dict] = None):
    """
    Load model + tokenizer for inference.
    Supports two checkpoint formats:
      A) Unified: torch.save({... 'tokenizer': <BPETokenizer>, 'config': {...}, 'model_state_dict': ...})
      B) State dict only: torch.save(model.state_dict(), ...)
         -> requires tokenizer_path (JSON saved via BPETokenizer.save)

    Returns: (model, tokenizer, dataset_meta)
    """
    device = get_device(device_override)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Case A: unified checkpoint (dict with tokenizer)
    if isinstance(ckpt, dict) and ("tokenizer" in ckpt or "model_state_dict" in ckpt):
        if "tokenizer" in ckpt:
            tokenizer = ckpt["tokenizer"]
        elif tokenizer_path:
            tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            raise RuntimeError(
                "Checkpoint appears to be unified but has no tokenizer. Provide tokenizer_path."
            )

        dataset_meta = DatasetMeta(tokenizer)

        cfg = ckpt.get("config", None)
        if config_override:
            if cfg is None:
                cfg = {}
            cfg.update(config_override)

        model = _build_model_for_tokenizer(tokenizer, dataset_meta, cfg)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model, tokenizer, dataset_meta

    # Case B: state-dict only
    if tokenizer_path is None:
        raise RuntimeError(
            "State-dict checkpoint detected. You MUST provide tokenizer_path saved via BPETokenizer.save()."
        )
    tokenizer = BPETokenizer.load(tokenizer_path)
    dataset_meta = DatasetMeta(tokenizer)

    model = _build_model_for_tokenizer(tokenizer, dataset_meta, config_override)
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model, tokenizer, dataset_meta


def generate_response(model: Transformer,
                      tokenizer: BPETokenizer,
                      dataset_meta: DatasetMeta,
                      user_text: str,
                      max_len: int = 50,
                      temperature: float = 1.0,
                      greedy: bool = True) -> str:
    """High-level helper for Streamlit."""
    text = normalize_urdu_text(user_text or "")
    if not text:
        return ""
    return generate_text(
        model, tokenizer, dataset_meta, text,
        max_len=max_len, temperature=temperature, greedy=greedy
    )
