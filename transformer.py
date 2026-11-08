import math

import torch
from torch import nn
from transformers import GPT2Tokenizer


class inputEmbeding(nn.Module):

    def __init__(self, d_model: int, voca_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = voca_size
        self.embeding = nn.Embedding(num_embeddings=voca_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeding(x)
        return x * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(dim=1)
        formula = pos * torch.exp(
            torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model
        )

        pe[:, 0::2] = torch.sin(formula)
        pe[:, 1::2] = torch.cos(formula)

        self.register_buffer("pe", pe.unsqueeze(dim=0))

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.size(1), :]
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.Norm = nn.LayerNorm(normalized_shape=d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Norm(x)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)

        self.layer1 = nn.Linear(in_features=input_dim, out_features=4 * input_dim)
        self.activation = nn.ReLU()

        self.layer2 = nn.Linear(in_features=4 * input_dim, out_features=input_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)

        x = self.layer2(x)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.MHA = nn.MultiheadAttention(num_heads=num_heads, embed_dim=d_model, dropout=dropout, batch_first=True)
    def forward(self, q, k, v) -> torch.Tensor:
        attn_output, _ = self.MHA(q, k, v, need_weights=False)
        return self.dropout(attn_output)

class CasualMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dropout: float, d_model: int) -> None:
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.CMHA = nn.MultiheadAttention(num_heads=num_heads, embed_dim=d_model, dropout=dropout, batch_first=True)
    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn_output, _ = self.CMHA(x, x, x, mask=mask, require_weights=False)
        return self.dropout(attn_output)



class Encoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, seq_len: int, num_heads: int, dropout: float) -> None:
        self.inputEmbeding = inputEmbeding(d_model=d_model, voca_size=vocab_size)
        self.positionalEncoding = PositionalEncoding(seq_len=seq_len, d_model=d_model)

        self.MHA = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.nrom1 = LayerNorm(d_model=d_model)

        self.ffn = FeedForwardNetwork(input_dim=d_model, dropout=dropout)
        self.norm2 = LayerNorm(d_model=d_model)
    def forward(self, x):
        x = self.inputEmbeding(x)
        x = self.positionalEncoding(x)

        x = self.norm1(x + self.MHA(x))
        x = self.norm2(x + self.ffn(x))

        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, num_heads: int, dropout: float, seq_len: int, encoder_output: torch.Tensor) -> None:
        super().__init__()
        self.outputEmbeding = inputEmbeding(d_model=d_model, voca_size=vocab_size)
        self.positionalEncoding =  PositionalEncoding(seq_len=seq_len, d_model=d_model)

        self.CMHA = CasualMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = LayerNorm(d_model=d_model)

        self.crossMultiHeadAttention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm2 = LayerNorm(d_model=d_model)

        self.ffn = FeedForwardNetwork(input_dim=d_model, dropout=dropout)
        self.norm2 = LayerNorm(d_model=d_model)
    def forward(self, x, encoder_output):
        shifted_x = x[:, 1:] #Crucial for the teacher forcing meathod
        x = self.outputEmbeding(shifted_x)
        x = self.positionalEncoding(x)

        x = self.norm1(x + self.CMHA(x))
        x = self.norm2(x + self.crossMultiHeadAttention(x, encoder_output, encoder_output))
        x = self.norm3(x + self.ffn(x))
        return x


### Defining the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
global_vocab_size = tokenizer.vocab_size


### Hyperparamaters
