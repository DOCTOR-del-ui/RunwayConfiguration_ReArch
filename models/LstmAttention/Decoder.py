import torch
import torch.nn as nn
from models.LstmAttention.Attn import MultiHeadAttentionLayer, FeedForward
from models.LstmAttention.Embdding import Embedding

class DecoderLayer(nn.Module):
    def __init__(self, n_head = 8, d_model = 512, ffn_hidden = 2048, dropout = 0.3, use_norm = True):
        super(DecoderLayer, self).__init__()
        self.use_norm = use_norm
        self.cross_attention = MultiHeadAttentionLayer(n_head, d_model)
        self.feedforward = FeedForward(d_model, ffn_hidden, dropout)
        self.layernorm1 = nn.LayerNorm(normalized_shape = d_model)
        self.layernorm2 = nn.LayerNorm(normalized_shape = d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, enc_out):
        out = self.cross_attention(x, enc_out, enc_out)
        out = self.dropout1(out)
        out = x + out
        if self.use_norm:
            out = self.layernorm1(out)
        out_c = out[:]
        out = self.feedforward(out)
        out = self.dropout2(out)
        out = out_c + out
        if self.use_norm:
            out = self.layernorm2(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self,  n_head = 8, d_model = 512, \
                        ffn_hidden = 1024, max_len = 512, \
                        fea_dim = 15, num_lstm_layers = 2, \
                        num_dec_layers = 1, dropout = 0.3, \
                        use_norm1 = True, use_norm2 = True, use_pos = True):
        super(Decoder, self).__init__()
        self.promptembedding = Embedding(fea_dim, d_model, num_lstm_layers, max_len, dropout, use_norm1, use_pos)
        self.decoderblocks = nn.ModuleList([
            DecoderLayer(n_head, d_model, ffn_hidden, dropout, use_norm2) for _ in range(num_dec_layers)
        ])
        
    def forward(self, x, enc_out):
        out = self.promptembedding(x)
        for layer in self.decoderblocks:
            out = layer(out, enc_out)
        return out
    
    

