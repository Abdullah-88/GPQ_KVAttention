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

class FFUnit_index(nn.Module):
    def __init__(self,dim,index_dim):

        super().__init__()

        self.proj =  nn.Linear(dim,index_dim,bias=False)
        self.modulate = VecDyGeluSine(index_dim)

    def forward(self, x):

        x = self.proj(x)

        u,v = x,x

        u = self.modulate(u)
        
        g = u * v

        return g

class SGPQ_KVAttention(nn.Module):
    def __init__(self, d_model, n_heads,top_k, index_dim=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k
        self.index_dim = index_dim
      
        self.GP_q = FFUnit(self.d_model)
        self.GP_kv = FFUnit(self.d_model)
        
        self.GP_q_idx = FFUnit_index(self.d_model,self.n_heads * self.index_dim)
        self.GP_k_idx = FFUnit_index(self.d_model,self.n_heads * self.index_dim)
        
    def forward(self, x):
       
        B, L, _ = x.shape
        H, D, idx_D = self.n_heads, self.d_head, self.index_dim
              
        q = self.GP_q(x).view(B, L, H, D).transpose(1, 2)  
        k = self.GP_kv(x).view(B, L, H, D).transpose(1, 2)
        v = self.GP_kv(x).view(B, L, H, D).transpose(1, 2)
             
        q_idx = self.GP_q_idx(x).view(B, L, H, idx_D).transpose(1, 2) 
        k_idx = self.GP_k_idx(x).view(B, L, H, idx_D).transpose(1, 2)
                    
        Total_L = k.shape[-2]
              
        idx_scores = torch.matmul(q_idx, k_idx.transpose(-1, -2))
        idx_scores = F.relu(idx_scores)
               
        current_k = min(self.top_k, Total_L)
              
        topk_scores, topk_indices = torch.topk(idx_scores, k=current_k, dim=-1, largest=True, sorted=False)
               
        gather_indices_kv = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)
              
        k_sparse = torch.gather(k.unsqueeze(2).expand(-1, -1, L, -1, -1), 3, gather_indices_kv)
        v_sparse = torch.gather(v.unsqueeze(2).expand(-1, -1, L, -1, -1), 3, gather_indices_kv)
            
        scores = torch.matmul(q.unsqueeze(-2), k_sparse.transpose(-1, -2)).squeeze(-2) 
        scores = scores / (D ** 0.5)
               
        attn_weights = F.softmax(scores, dim=-1)
               
        context = torch.matmul(attn_weights.unsqueeze(-2), v_sparse).squeeze(-2) # (B, H, L, D)
               
        context = context.transpose(1, 2).contiguous().view(B, L, H * D)
               
        return context

class SGPQ_KVAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):

        super().__init__()

        self.norm_1 =  VecDyT(dim)
        self.norm_2 =  VecDyT(dim)
        self.attn = SGPQ_KVAttention(dim,num_heads,5)
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

class SGPQ_KVAttentionTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()

        self.model = nn.Sequential(
            *[SGPQ_KVAttentionBlock(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):

        return self.model(x)
