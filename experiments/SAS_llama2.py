from transformers import AutoTokenizer    #Not really needed yet
from transformers import LlamaForCausalLM
import transformers
import torch
import torch.optim as optim

import sys
sys.path.append('../')
from model_SAS import *
from model import Block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name) #Not really needed yet
model = LlamaForCausalLM.from_pretrained(model_name)

llama_layer = model.model.layers[2].to(device)

#my_tensor_input = torch.rand((3, 1024, 4096))
#with torch.no_grad():
#    out = llama_layer( my_tensor_input )[0]

@dataclass
class LLamaBlockConfig:
    '''
     From LLama2 config:
         {"dim"          : 4096, 
          "multiple_of"  : 256, 
          "n_heads"      : 32, 
          "n_layers"     : 32, 
          "norm_eps"     : 1e-05, 
          "vocab_size"   : -1}
    '''
    block_size: int = 3192 
    vocab_size: int = 32000 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 32 
    n_head: int = 32 
    n_embd: int = 4096 
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    transformer_block_type: str = 'SASP' # SASP, PreLN
    use_v: bool = False  # use V in SASP or not
    use_proj: bool = False # use the output projection in SASP or not
    llama_mlp: bool = True # use the llama form of the feed forward network

config = LLamaBlockConfig()
new_block = SimplifiedTransformerBlock(config).to(device)
#new_block = Block(config).to(device)

N_train = 25000 
batch_size = 1 
n_embd = config.n_embd
block_size = config.block_size

optimizer = optim.RMSprop(new_block.parameters(), lr=0.00025)
criterion = nn.MSELoss()

for train_step in range(N_train):
    # Generate training data using llama_layer
    x = torch.randn(batch_size, block_size, n_embd, requires_grad=True).to(device)
    # since both models perform normalization as the first step of 
    # each block, we need to compare normalized outputs
    x = F.normalize(x, dim=2)
    with torch.no_grad():
        y = llama_layer(x)[0].to(device)
        y = F.normalize(y, dim=2)

    # Forward pass, with normalization
    output = new_block(x)
    output = F.normalize(output, dim=2)

    # Compute the cosine similarity loss
    loss =  - torch.mean(torch.mean(torch.sum(y*output, dim=2), dim=1))

    # Backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for monitoring training progress
    print(f"Training Step: {train_step + 1}, Loss: {loss.item()}")

