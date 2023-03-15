import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels:int=3,patch_size:int=16,emb_size:int = 768,img_size:int=256) -> None:
        self.patch_size =  patch_size

        super(PatchEmbedding,self).__init__()

        self.projection = nn.Conv2d(in_channels,emb_size,patch_size,patch_size)

        self.cls_token = nn.Parameter(torch.rand(1,1,emb_size))

        self.position_encoding = nn.Parameter(torch.randn(img_size // patch_size) ** 2 + 1,emb_size))

    def forward(self,x:torch.Tensor):

        x = self.projection(x) # b,emb_size,patch_size,patch_size
        
        b,e,w,h = x.shape

        x = x.view(b,w*h,e)

        cls_token = torch.repeat_interleave(self.cls_token,b,dim=0)

        # pre-pend the cls token to the input

        x = torch.cat([cls_token,x],dim=1)

        # add the positional embedding

        return x + self.position_encoding

class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size:int = 512,num_heads:int = 8,dropout:float = 0.0) -> None:
        super().__init__()
        self.emb_size = emb_size

        self.num_heads = num_heads

        self.head_dim = emb_size // num_heads

        self.keys = nn.Linear(emb_size,emb_size)

        self.queries= nn.Linear(emb_size,emb_size)

        self.values= nn.Linear(emb_size,emb_size)

        self.att_drop = nn.Dropout(dropout)

        self.projection = nn.Linear(emb_size,emb_size)

    def forward(self,x:torch.Tensor,mask:torch.Tensor = None) -> torch.Tensor:
        # query, key and values
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        b,seq_len,emb_size = queries.shape
        # split q,k,v in num_heads
        queries = queries.view(b,self.num_heads,seq_len,self.head_dim)

        keys = keys.view(b,self.num_heads,seq_len,self.head_dim)

        values = values.view(b,self.num_heads,seq_len,self.head_dim)

        energy = torch.einsum('bhqd,bhkd -> bhqk',queries,keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask,fill_value)
        
        scaling = self.emb_size ** 0.5
        attention = torch.softmax(energy, dim=-1) / scaling
        attention = self.att_drop(attention)

        # multiply attention with values
        out = torch.einsum('bhad, bhdv -> bhav',attention,values)

        out = out.view(b,seq_len,self.num_heads * self.head_dim)



if __name__ == "__main__":
    x = torch.randn(4,3,224,224)

    patch_embedd = PatchEmbedding()

    y = patch_embedd(x)