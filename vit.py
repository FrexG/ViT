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

if __name__ == "__main__":
    x = torch.randn(4,3,224,224)

    patch_embedd = PatchEmbedding()

    y = patch_embedd(x)