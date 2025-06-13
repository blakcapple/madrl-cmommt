
import torch.nn as nn
import torch 
import torch.nn.functional as F  

class SelfAtten(nn.Module):
    '''self attention layer'''
    def __init__(self, config):
        super().__init__()
        self.atten_input_dims = 8
        self.query_dims = config['atten_query_dims']
        self.value_dims = config['atten_value_dims']
        self.key_dims = config['atten_key_dims']
        self.atten_head = config['atten_heads']
        self.embed_dim = config['atten_embed_dims']
        self.query = nn.Linear(self.atten_input_dims, self.query_dims)
        self.value = nn.Linear(self.atten_input_dims, self.value_dims)
        self.key = nn.Linear(self.atten_input_dims, self.key_dims)
        self.multiattention = nn.MultiheadAttention(self.embed_dim, self.atten_head, batch_first=True)
        self.act = nn.ReLU()
    
    def forward(self,input, mask=None):
        proj_query = self.query(input)
        proj_key = self.key(input)
        proj_value = self.value(input)
        output, _ = self.multiattention(proj_query, proj_key, proj_value, mask)
        
        return self.act(output[:, 0])
    

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8):

        super().__init__()

        self.emb = emb
        self.heads = heads

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask=None):
        
        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if mask is not None:
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            dot = dot.view(b,h,t,t)
            dot = dot.masked_fill(mask == 0, -1e9)
            dot = dot.view(b*h,t,t)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)
