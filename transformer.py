import torch
from torch import nn
import math

class inputEmbeding(nn.Module):
    def __init__(self, d_model: int, voca_size: int):
        self.d_model = d_model
        self.vocab_size = voca_size
        self.embeding = nn.Embedding(num_embeddings=voca_size, embedding_dim=d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeding(x)
        return x * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        self.seq_len = seq_len
        self.d_model = d_model
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.long)
        formula = pos * torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)

        pe[:, 0::2] = torch.sin(formula)
        pe[:, 1::2] = torch.cos(formula)

        self.register_buffer("pe", pe.unsqueeze(dim=0))
    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, :x.size(1), :]
        return x






class Encoder(nn.Module):
    def __init__(self, ) -> None:
        self.inputEmbeding = InputEmbeding()
        self.positionalEncoding = PositionalEncoding()

        self.MHA = MultiHeadAttention()
        self.nrom1 = LayerNorm()

        self.ffn = FeedForwardNetwork()
        self.norm2 = LayerNorm()
    def forward(self, x) -> torch.Tensor:
        pass
