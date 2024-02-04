
import math
#import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class SASLayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SASMLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.llama_mlp = getattr( config, 'llama_mlp', False)
        if self.llama_mlp:
           N = int( 2.1/3. * 4 * config.n_embd )
           self.c_mask  = nn.Linear(config.n_embd, N, bias=config.bias) # LLama2
        else:
           N = int( 4 * config.n_embd )

        self.c_fc    = nn.Linear(config.n_embd, N, bias=config.bias)
        self.actv    = nn.GELU()
        self.c_proj  = nn.Linear( N, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        

    def forward(self, x):
        if self.llama_mlp:
           xtmp = self.c_mask(x)  # TODO:  testing. Remove this -- it'sLLama Like, not shaped-attention like
        x = self.c_fc(x)
        x = self.actv(x)
        if self.llama_mlp:
           x = x * xtmp           # TODO:  testing. Remove this -- it'sLLama Like, not shaped-attention like
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalShapedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Provide the option to use or exclude the projection matrix
        self.use_proj = getattr( config, 'use_proj', False)
        # Provide the option to use Q,K,V or just Q,K
        self.use_v    = getattr( config, 'use_v'   , False)

        # Allocate output projection, if designated
        if self.use_proj:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Allocate QKV or just QK parameters in the attention
        self.num_var_to_pack = (3 if self.use_v else 2) 
        self.c_attn = nn.Linear(config.n_embd, self.num_var_to_pack * config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Maximum number of tokens to process.  Pre-alocate mask and attention component.
        self.max_block_size = config.block_size

        # Parameters specific to the shaped attention
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.bfloat16))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.bfloat16))
        self.custom_variable_initialization()

    def custom_variable_initialization(self):
        # shaped attention has parameter initialization conditions
        with torch.no_grad():
            self.alpha.fill_(1.0)
            self.beta.fill_(0.1)
            self.gamma.fill_(0.1)
            self.c_attn.weight[self.n_embd:, :].fill_(0.0)

        self.register_buffer("MC", F.softmax(1e20 * torch.tril(torch.ones(self.max_block_size, self.max_block_size)), dim=-1, dtype=torch.bfloat16).view(1, 1, self.max_block_size, self.max_block_size))
        self.register_buffer("Id", F.softmax(torch.eye(self.max_block_size), dim=-1, dtype=torch.bfloat16).view(1, 1, self.max_block_size, self.max_block_size))

    def forward(self, x):
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        B, T, C = x.size()

        assert T <= self.max_block_size

        q, k, *v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Pass x directly if no V is used in the attention
        v = v[0] if self.use_v else x
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.MC[:, :, :T, :T] == 0, float('-1e20'))

        Id = self.Id[:, :, :T, :T].expand(B, self.n_head, T, T)
        MC = self.MC[:, :, :T, :T].expand(B, self.n_head, T, T)
        att = beta * F.softmax(att, dim=-1)
        att = att + alpha * Id - gamma * MC

        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection, if enabled
        if self.use_proj:
            y = self.c_proj(y)

        return y


class SimplifiedTransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = SASLayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalShapedAttention(config)
        self.mlp = SASMLP(config)
        self.beta_SA = nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))
        self.beta_FF = nn.Parameter(torch.tensor(0.2, dtype=torch.bfloat16))
        self.initialize_parameters()

    def initialize_parameters(self):
        # The simplified transformer architecture has some requirements on 
        # initialization values that are critical to the architecture.
        #  (1) beta_FF < beta SA = 1.0
        #
        #  (2) The W_K part of attn.c_attn.weight = 0
        #  (3) attn.alpha = 1.0
        #  (4) attn.beta = attn.gamma, so that the initial attn matrix is the identity
        #      attn.beta = attn.gamma = 1.0 is acceptable
        self.attn.custom_variable_initialization()
        with torch.no_grad():
            self.beta_SA.fill_(1.0)
            self.beta_FF.fill_(0.2)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.beta_SA * self.attn(x) + self.beta_FF * self.mlp(x)
        # Because the result is normalized, the beta_SA and beta_FF only
        # have one effective parameter between them, as can be verified 
        # by using the following line instead of the one above:
        #
        #      x = (1 - self.beta_FF) * self.attn(x) + self.beta_FF * self.mlp(x)
        #
        # The same logic applies to alpha in the Shaped Attention class.
        return x


###############################################################################
#
#
#
###############################################################################

