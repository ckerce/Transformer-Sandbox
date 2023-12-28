
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb  # Import the pdb module


class CausalShapedAttention(nn.Module):
    '''
    nano-GPT style code to implement causal shaped attention and apply it to an input.

    Pseudocode with bad dimensionality:

    self.W_Q = nn.Linear(self.head_dim, self.head_dim, bias=False)
    nn.init.zeros_(self.W_Q.weight)
    self.W_K = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, X):
        seq_len = X.size(1)  # Assuming X is of shape [batch, seq_len, token_embd_dim]
        M_tmp = self.M[:seq_len, :seq_len]
        C_tmp = self.C[:seq_len, :seq_len]
        # Split X into self.heads chunks along the token_embd_dim dimension
        X_chunks = torch.chunk(X, self.heads, dim=2)

        # Process each chunk and store the results
        processed_chunks = []
        for X_h in X_chunks:
            Q_h = self.W_Q(X_h)
            K_h = self.W_K(X_h)
            QKT = torch.bmm(Q_h, K_h.transpose(2,1)) / self.scale
            A_h = nn.functional.softmax( QKT, dim=-1) + M_tmp
            Id = torch.eye( X_h.size(1) )
            modified_X_h = self.alpha * X_h + (self.beta * A_h - self.gamma * C_tmp) * X_h
            processed_chunks.append(modified_X_h)

        # Concatenate the processed chunks back together
        X_out = torch.cat(processed_chunks, dim=2)
        return X_out


    '''

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key & query projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        #self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.custom_variable_initialization()

    def custom_variable_initialization(self):
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        # Initialize W_K to zero.
        W_QK = self.c_attn.weight.detach()
        W_QK[self.n_embd:, :] = 0
        self.c_attn.weight = torch.nn.Parameter( W_QK )


        # Manually create buffers for attention components
        self.register_buffer("M", torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1))
        self.MC = F.softmax(1e20*self.M, dim=-1)

        # Manually create buffers for attention components
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
    
        B, T, C = x.size()
    
        pdb.set_trace()  # Add a breakpoint here to start debugging

        q, k  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-1e20'))
    
        Id = torch.eye(T).view(1,1,T,T).expand(B,self.n_head,T,T)
        MC = self.MC.view(1,1,T,T).expand(B, self.n_head, T, T)
        #Id = torch.eye(T).unsqueeze(0).unsqueeze(1).expand(B, self.n_head, T, T)
        #MC = self.MC.unsqueeze(0).unsqueeze(1).expand(B, self.n_head, T, T)
        att = beta * F.softmax(att, dim=-1)
        att = att + alpha * Id  - gamma * MC
    
    
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
    
        # NOTE:  There is no skip connection, so this is the output of the
        # Shaped Attention applied to the input.

        return y


class MakeConfig:
    def __init__(
        self,
        n_embd=128,           # Embedding dimension
        n_head=4,            # Number of attention heads
        dropout=0.1,          # Dropout probability
        bias=False,           # Whether to include bias in linear layers
        block_size=256,      # Maximum sequence length
    ):
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.block_size = block_size


config = MakeConfig()
tmp = CausalShapedAttention( config )


