import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class InputProjection(nn.Module):
    def __init__(self, input_dim=48, embed_dim=256):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.linear(x)  
        x = self.activation(x)
        return self.layer_norm(x)  
    

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=256, max_len=100):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.xavier_uniform_(self.pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]  
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.w_q(x) 
        K = self.w_k(x)
        V = self.w_v(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embed_dim)

        return self.w_o(attention_output), attention_weights
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_output, weights = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x, weights
    
class TransformerTemporalEncoder(nn.Module):
    def __init__(self, input_dim=48, embed_dim=64, num_heads=4, 
                 num_layers=3, ff_dim=256, dropout=0.1):
        super().__init__()
        
        self.input_proj = InputProjection(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=100)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attention_weights = []
        
        x = self.input_proj(x)  
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x, weights = block(x)
            attention_weights.append(weights)
        
        return x, torch.stack(attention_weights)
    

if __name__ == '__main__':
    from tqdm import tqdm
    device = "cuda"
    transformer = TransformerTemporalEncoder().to(device)
    batch_size = 1024
    
    for _ in tqdm(range(100)):
        data = torch.rand(batch_size, 64, 48).to(device)
        temporal_encoded, attention_weights = transformer(data)
    
