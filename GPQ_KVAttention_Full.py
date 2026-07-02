import torch
from torch import nn, Tensor
import torch.nn.functional as F



class VecDyT(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x


class VecDyGeluSine(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))
        self.beta = nn.Parameter(torch.randn(input_shape))
        self.gamma = nn.Parameter(torch.randn(1))
        self.etta = nn.Parameter(torch.randn(1))
        self.gelu = nn.GELU()

    def forward(self, x):



        x = self.gamma * self.gelu(self.alpha * x) + self.etta * torch.sin(self.beta * x)

        return x

class FFUnit(nn.Module):
    def __init__(self,dim):

        super().__init__()

        self.proj =  nn.Linear(dim,dim,bias=False)
        self.modulate = VecDyGeluSine(dim)


    def forward(self, x):

        u, v = x, x

        u = self.modulate(u)
        v = self.proj(v)
        g = u * v

        return g




class GPQ_KVAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
     
      
        self.GP_q = FFUnit(d_model)
        self.GP_kv = FFUnit(d_model)
     
       
        
    def forward(self, x):
       
        batch_size = x.size(0)

       
        query = self.GP_q(x)
        keyvalue = self.GP_kv(x)
       
       
     
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = keyvalue.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = keyvalue.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

     
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)

       

       
        attention_weights = F.softmax(attention_scores, dim=-1)
        

     
        out = torch.matmul(attention_weights, value)

       
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

       
        return out




class GPQ_KVAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):

        super().__init__()

        self.norm_1 =  VecDyT(dim)
        self.norm_2 =  VecDyT(dim)
        self.attn = GPQ_KVAttention(dim,num_heads)
        self.feedforward = FFUnit(dim)


    def forward(self, x):


        residual = x

        x = self.norm_1(x)

        x = self.attn(x)

        x = x + residual

        residual = x

        x = self.norm_2(x)

        x = self.feedforward(x)

        x = x + residual

        return x


class GPQ_KVAttentionTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()

        self.model = nn.Sequential(
            *[GPQ_KVAttentionBlock(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):

        return self.model(x)

