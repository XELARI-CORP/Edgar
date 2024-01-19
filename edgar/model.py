import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np



class ConvLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b = torch.nn.parameter.Parameter(torch.zeros((dim,1)), requires_grad=True)
        self.a = torch.nn.parameter.Parameter(torch.ones((dim,1)), requires_grad=True)
        self.eps = 1e-6
        
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        dif = x - mean
        var = dif.pow(2).mean(1, keepdim=True)
        x = dif / torch.sqrt(var + self.eps)
        x = self.a*x + self.b
        
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        
        self.C1 = nn.Conv1d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.C2 = nn.Conv1d(ch, ch, kernel_size=3, stride=1, padding=1)
        torch.nn.init.kaiming_uniform_(self.C1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.C2.weight, gain=1.0)
        
        self.LN = ConvLayerNorm(ch)
        self.drop = nn.Dropout(0.1)
        
        
    def forward(self, x):
        c = self.C1(x)
        c = F.relu(c)
        c = self.drop(c)
        c = self.C2(c)
        
        c = c + x
        c = self.LN(c)
        
        return c
    
        
class MHAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        
        self.heads = heads
        self.dim = dim
        self.depth = dim//heads
        self.norm = np.sqrt(self.depth)
        
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.O = nn.Linear(dim, dim)
        
        for l in (self.Q, self.K, self.V, self.O):
            torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
            torch.nn.init.zeros_(l.bias)
        
        
    def forward(self, x, mask):
        seq = x.shape[1]
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # batch, seq, heads, dim -> batch, heads, seq, dim
        q = q.view(-1, seq, self.heads, self.depth).permute(0, 2, 1, 3)
        k = k.view(-1, seq, self.heads, self.depth).permute(0, 2, 3, 1)
        v = v.view(-1, seq, self.heads, self.depth).permute(0, 2, 1, 3)
        
        #att
        g = torch.matmul(q, k)
        g /= self.norm
        
        if mask is not None:
            g -= (mask*1e9)
        A = F.softmax(g, dim=-1)

        att = torch.matmul(A, v)# b,h,s,d

        att = att.permute(0, 2, 1, 3)# b,s,h,d
        att = torch.reshape(att, (att.shape[0], att.shape[-3], self.dim))
        att = self.O(att)
        
        return att
    
    
class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, do):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.do = do
        
        self.Att = MHAttention(self.dim, self.heads)

        self.drop1 = nn.Dropout(self.do)
        self.drop2 = nn.Dropout(self.do)

        self.LN1 = nn.LayerNorm(normalized_shape=dim)
        self.LN2 = nn.LayerNorm(normalized_shape=dim)

        self.FC1 = nn.Linear(dim, dim*4)
        self.FC2 = nn.Linear(dim*4, dim)
        torch.nn.init.kaiming_uniform_(self.FC1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.FC2.weight, gain=1.0)


    def forward(self, x, mask):
        att = self.Att(x, mask)
        att = self.drop1(att)
        
        x = att + x
        x = self.LN1(x)

        d = self.FC1(x)
        d = F.gelu(d)
        d = self.FC2(d)
        d = self.drop2(d)

        x = d + x
        x = self.LN2(x)

        return x
    
    
class ComplementaryGraphLayer(nn.Module):
    def __init__(self, dim, do):
        super().__init__()
        self.dim = dim
        self.do = do
        
        self.FC1 = nn.Linear(dim+1, dim*2)
        self.FC2_compl = nn.Linear(dim*2, dim)
        self.FC2_noncompl = nn.Linear(dim*2, dim)
        torch.nn.init.kaiming_uniform_(self.FC1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.FC2_compl.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.FC2_noncompl.weight, gain=1.0)
        
        self.drop = nn.Dropout(self.do)
        self.LN = nn.LayerNorm(normalized_shape=dim)


    def take_vectors_by_indexes(self, x, compl_idx):
        b, seq, d = x.shape
        
        compl_idx = compl_idx[..., None].repeat((1,1,d))
        compl_idx *= d
        compl_idx += torch.arange(d, device=x.device)
        compl_idx = compl_idx.view(b, -1)
        
        x = x.view(b, -1)
        y = torch.gather(x, 1, compl_idx)
        y = y.view(b, seq, d)
        
        return y
        
        
    def forward(self, args):
        x, struct = args
        
        compl_mask = (struct!=-1.).float() # b, seq (helix nas)
        non_compl_mask = (1. - compl_mask) # (free nas)
        
        # complementary indexes + non complementary indexes
        compl_idx = struct*compl_mask + torch.arange(x.shape[1], device=x.device)*non_compl_mask
        compl_idx = compl_idx.long()
        
        compl_mask = compl_mask[..., None] # b, seq, 1
        non_compl_mask = non_compl_mask[..., None]
        
        # first linear
        a = torch.cat([x, compl_mask], dim=-1) # b, seq, dim -> b, seq, dim+1
        a = self.FC1(a)
        compl_a = self.take_vectors_by_indexes(a, compl_idx)
        a = a+compl_a
        a = F.relu(a)
        a = self.drop(a)
        
        # second linear
        a_compl = self.FC2_compl(a)*compl_mask
        a_noncompl = self.FC2_noncompl(a)*non_compl_mask
        a = a_compl + a_noncompl
        x = self.LN(a)
        
        return x
    
    
class NaEmbedding(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        
        self.linear = nn.Linear(vocab+4, dim)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.zeros_(self.linear.bias)
        
        self.LN = nn.LayerNorm(normalized_shape=dim)
        
        
    def forward(self, seq, na_type, mean_mask):
        batch_size, slen = seq.shape
        
        nbs = torch.nn.functional.one_hot(seq.long(), num_classes=(self.vocab+2))
        nbs[:, 0, -2] = 1 # begin of sequence tag
        first_pad_idx = torch.sum(mean_mask, dim=1)
        last_nb_idx = first_pad_idx - 1
        nbs[torch.arange(batch_size).long(), last_nb_idx.long(), -1] = 1 # end of sequence tag
        
        typ = torch.nn.functional.one_hot(na_type.long(), num_classes=2) # b -> b, 2
        typ = typ.view(-1, 1, 2).repeat((1, slen, 1)) # b, 2 -> b, seq, 2
        
        nbs = torch.cat([nbs, typ], dim=-1) # b, seq, vocab+4
        x = self.linear(nbs.float()) # b, seq, vocab+4 -> b, seq, dim
        x = self.LN(x)
        
        return x


class Edgar(nn.Module):
    
    DEFAULT_SEQ_LEN = 256
    
    def __init__(self, 
                 vocab, 
                 dim, 
                 conv_layers, 
                 transformer_layers, 
                 heads):
        
        super().__init__()
        self.dim = dim
        self.vocab = vocab
        
        self.embedding = NaEmbedding(vocab, dim)
        
        # Aggregator
        self.Agg = torch.nn.parameter.Parameter(torch.empty(dim, dtype=torch.float32, requires_grad=True))
        torch.nn.init.normal_(self.Agg, mean=0.0, std=1.0)
        
        # conv
        self.conv_blocks = nn.ModuleList()
        for i in range(conv_layers):
            self.conv_blocks.append(ConvBlock(dim))
        
        # graph
        self.graph_layer = ComplementaryGraphLayer(dim=dim, do=0.1)
        
        # transformer
        self.PE = None
        self.transformer_layers = nn.ModuleList()
        for i in range(transformer_layers):
            self.transformer_layers.append(EncoderLayer(dim=dim, heads=heads, do=0.1))
        
        # OUT    
        self.out = nn.Linear(dim, 2)
        torch.nn.init.xavier_uniform_(self.out.weight, gain=1.0)
        torch.nn.init.zeros_(self.out.bias)
        
        
    def calculate_pos_enc(self, seq, dim):
        pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] 
            if pos != 0 else np.zeros(dim) 
            for pos in range(seq)
        ], dtype=np.float32)

        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1

        return pos_enc
    
    
    def get_pos_enc(self, seq):
        slen = seq.shape[1]
        
        if self.PE is None:
            pe = self.calculate_pos_enc(seq=(self.DEFAULT_SEQ_LEN+1), dim=self.dim)[1:]
            self.PE = torch.from_numpy(pe).to(seq.device)
        
        if (slen>self.PE.shape[0]):
            pe = self.calculate_pos_enc(seq=(slen+1), dim=self.dim)[1:]
            self.PE = torch.from_numpy(pe).to(seq.device)
            return self.PE
        
        elif slen<self.PE.shape[0]:
            PE = self.PE[:slen]
            return PE
        
        return self.PE
        
        
    def forward(self, seq, struct, na_type):
        batch_size, slen = seq.shape
        
        pad_mask = (seq==0.).float()
        att_mask = torch.cat(
            [torch.zeros((batch_size, 1), dtype=torch.float32, device=seq.device), pad_mask], 
            dim=1) # -> b, 1+seq
        mean_mask = 1 - pad_mask
        PE = self.get_pos_enc(seq)
        
        # EMB
        x = self.embedding(seq, na_type, mean_mask)
        
        # Conv
        x = torch.permute(x, (0, 2, 1)).contiguous() # b, seq, dim -> b, dim, seq
        for l in self.conv_blocks:
            x = l(x)
        x = torch.permute(x, (0, 2, 1)).contiguous() # b, dim, seq -> b, seq, dim
        
        # Graph
        x += PE
        x = self.graph_layer((x, struct))
            
        # Transformer
        agg = self.Agg[None, None, :].repeat((batch_size, 1, 1)) # dim -> b, 1, dim
        x = torch.cat([agg, x], dim=1) # b, seq, dim -> b, 1+seq, dim
        
        for l in self.transformer_layers:
            x = l(x, att_mask[:, None, None])
           
        # Readout
        agg, x = x[:, 0], x[:, 1:]
        
        preds = self.out(x) # -> b, seq, 2
        mean_mask /= torch.sum(mean_mask, dim=1)[..., None]
        preds = preds*mean_mask[..., None]
        
        out = torch.sum(preds, dim=1) # b, seq, 2 -> b, 2
        
        return out, agg
    
    
    
    
    
    
    
    
    
    
    
        