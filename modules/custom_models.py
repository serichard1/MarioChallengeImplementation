from .base_models import mobilenet_v3_small
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualVision(nn.Module):
    def __init__(
        self,
        backbone,
        in_size,
        nclasses=3,
        drop_ratio_head=0.4
    ):
        super().__init__()

        self.backbone = backbone
        self.drop_ratio_head = drop_ratio_head

        self.merge_bscans = self._create_sequential([in_size*2, 1024, 256, 96])
        self.hidden32 = nn.Sequential(nn.Linear(96, 32), nn.SiLU(inplace=True))
        self.merge_numeric = nn.Sequential(nn.Linear(32+5, 32), nn.SiLU(inplace=True))
        self.head =nn.Sequential(nn.Linear(32, 16), nn.Linear(16, nclasses))
        
    def _create_sequential(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=self.drop_ratio_head))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, bscan_ti, bscan_tj, side_eye, bscan_num, sex, age, delta_h):
        bscan_ti, bscan_tj = map(lambda f: self.backbone(f), (bscan_ti, bscan_tj))
        side_eye, bscan_num, sex, age, delta_h = map(lambda f: f.unsqueeze(1), (side_eye, bscan_num, sex, age, delta_h))

        bscans_embed = self.merge_bscans(torch.cat((bscan_ti, bscan_tj), dim=1))
        hidden32 = self.hidden32(bscans_embed)
        final_embed = self.merge_numeric(torch.cat((hidden32, side_eye, bscan_num, sex, age, delta_h), dim=1))

        logits = self.head(final_embed)
        
        return logits
    
class DualVision_student(nn.Module):
    """
    FOR TASK2 -- distillation, studen network without bscan at t+1
    """
    def __init__(
        self,
        backbone,
        in_size,
        nclasses=3,
        drop_ratio_head=0.4
    ):
        super().__init__()

        self.backbone = backbone
        self.mobile = mobilenet_v3_small()
        self.drop_ratio_head = drop_ratio_head

        self.merge_bscans = self._create_sequential([in_size, 1024, 1024, 512, 256, 128, 96])
        self.local_embed = self._create_sequential([1024, 512, 256, 96])

        self.hidden32 = nn.Sequential(nn.Linear(96*2, 32), nn.SiLU(inplace=True))
        self.merge_numeric = nn.Sequential(nn.Linear(32+5, 32), nn.SiLU(inplace=True))
        self.head =nn.Sequential(nn.Linear(32, 16), nn.Linear(16, nclasses))
        
    def _create_sequential(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=self.drop_ratio_head))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, bscan_ti, side_eye, bscan_num, sex, age, delta_h, localizer):
        bscan_ti = self.backbone(bscan_ti)
        localizer_embed = self.local_embed(self.mobile(localizer))

        side_eye, bscan_num, sex, age, delta_h = map(lambda f: f.unsqueeze(1), (side_eye, bscan_num, sex, age, delta_h))

        bscans_embed = self.merge_bscans(bscan_ti)
        hidden32 = self.hidden32(torch.cat((bscans_embed, localizer_embed), dim=1))
        final_embed = self.merge_numeric(torch.cat((hidden32, side_eye, bscan_num, sex, age, delta_h), dim=1))
        
        return final_embed
    

class CrossSightv5(nn.Module):
    """ Ensemble model """
    def __init__(
        self,
        nclasses=3,
        dropout_head=0.3
    ):
        super().__init__()

        self.attn = MultiheadAttention(input_dim=128, embed_dim=768, num_heads=12)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.dropout_head = dropout_head
    
        self.reducing = self._create_sequential([768, 512, 256, 64])
        self.to_hidden = nn.Linear(128, 768)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.SiLU(inplace=True), nn.Linear(32, nclasses))
        
    def _create_sequential(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, embeddings):

        attention_weighted = self.attn(embeddings)
        attention_weighted = self.avgpool(torch.permute(attention_weighted, (0, 2, 1)))

        embed = self.reducing(attention_weighted.squeeze(-1))
        logits = self.head(embed)

        return logits
    
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values = self.scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o

