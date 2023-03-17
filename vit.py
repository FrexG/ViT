import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels:int=3,patch_size:int=16,emb_size:int = 768,img_size:int=256) -> None:
        self.patch_size =  patch_size

        super(PatchEmbedding,self).__init__()

        self.projection = nn.Conv2d(in_channels,emb_size,patch_size,patch_size)

        self.cls_token = nn.Parameter(torch.rand(1,1,emb_size))

        self.position_encoding = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1,emb_size))

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
    def __init__(self,emb_size:int = 512,num_heads:int = 8,drop_out:float = 0.0) -> None:
        super().__init__()
        print(type(drop_out))
        self.emb_size = emb_size

        self.num_heads = num_heads

        self.head_dim = emb_size // num_heads

        self.keys = nn.Linear(emb_size,emb_size)

        self.queries= nn.Linear(emb_size,emb_size)

        self.values= nn.Linear(emb_size,emb_size)

        self.att_drop = nn.Dropout(drop_out)

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

        return out
    
class MLP(nn.Module):
    def __init__(self,emb_size:int = 768,expansion:int = 4,drop_out:float = 0.0):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size,expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(expansion * emb_size,emb_size)
        )
    def forward(self,x):
        return self.feed_forward(x)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,emb_size:int = 768,drop_out:float = 0.0
                 ,forward_expansion:int = 4,forward_drop_out:float = 0.0,):
        super().__init__()
        
        self.attention = nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size,drop_out=drop_out),
                nn.Dropout(drop_out)
            )
        self.feed_forward = nn.Sequential(
                nn.LayerNorm(emb_size),
                MLP(emb_size=emb_size,expansion=forward_expansion,drop_out=forward_drop_out),
                nn.Dropout(drop_out)
            )
    def forward(self, x):
        x_att = x + self.attention(x)
        out = x_att + self.feed_forward(x_att)
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self,depth=4):
        super().__init__()
        self.transformer = nn.ModuleList([TransformerEncoderBlock() for i in range(depth)])

    def forward(self,x):
        for t in self.transformer:
            x = t(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self,emb_size:int = 768,n_classes:int = 3):
        super().__init__()

        self.clf = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size,n_classes)
        )
        
    def forward(self,x):
        x = self.clf(torch.mean(x,dim=1))
        return x


class ViT(nn.Module):
    def __init__(self,emb_size:int = 768,n_classes:int = 3,img_size:int = 224,depth:int = 4) -> None:
        super().__init__()
        self.patch_embedd = PatchEmbedding(img_size=img_size)
        self.transformer_encoder = TransformerEncoder(depth)
        self.clf_head = ClassificationHead(emb_size,n_classes=n_classes)

    def forward(self,x):
        x = self.transformer_encoder(self.patch_embedd(x))
        x = self.clf_head(x)
        return x
        


if __name__ == "__main__":
    x = torch.randn(4,3,224,224)

    vit = ViT(img_size = 224)

    y = vit(x)
    print(y.shape)
