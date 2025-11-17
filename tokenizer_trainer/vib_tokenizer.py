import torch.nn as nn

class VibrationTokenizer(nn.Module):
    def __init__(self, vib_ae, token_embed_dim, freeze_encoder=True, embedding_dim=768):
        super().__init__()
        
        self.vib_encoder = vib_ae
        self.device = next(self.vib_encoder.parameters()).device
        self.dtype = next(self.vib_encoder.parameters()).dtype

        if freeze_encoder:
            for param in self.vib_encoder.parameters():
                param.requires_grad = False

        self.alignment_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=int(embedding_dim*2)
            ),
            nn.Sigmoid(),
            nn.Linear(
                in_features=int(embedding_dim*2),
                out_features=token_embed_dim
            )
        )

    def forward(self, x):

        class_embedding = self.vib_encoder.encode(x)

        z = self.alignment_layer(class_embedding)

        return z

import torch
import torch.nn.functional as F
import lightning as L
from torch import optim

class SubwordPooler(nn.Module):
    def __init__(self, d, mode="attn"):  # "mean" | "first" | "attn" | "gated"
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.q = nn.Parameter(torch.randn(d))   # learnable query
            self.tau = nn.Parameter(torch.tensor(0.07))  # optional temp
        elif mode == "gated":
            self.scorer = nn.Sequential(
                nn.Linear(d, d//4), nn.GELU(),
                nn.Linear(d//4, 1)
            )

    def forward(self, E, mask): 
        """
        E: (B, S, d) token embeddings  (pad 포함)
        mask: (B, S) 1 for real, 0 for pad
        return: (B, d)
        """
        B, S, d = E.shape
        mask = mask.float()
        if self.mode == "mean":
            w = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)  # (B,S)
            x = (E * w.unsqueeze(-1)).sum(dim=1)
        elif self.mode == "first":
            # 각 배치에서 첫 실토큰 인덱스 선택
            idx = mask.argmax(dim=1)  # 첫 1의 위치
            x = E[torch.arange(B), idx]
        elif self.mode == "gated":
            scores = self.scorer(E).squeeze(-1)              # (B,S)
            scores = scores.masked_fill(mask == 0, -1e30)
            w = torch.softmax(scores, dim=1)                 # (B,S)
            x = (E * w.unsqueeze(-1)).sum(dim=1)
        elif self.mode == "attn":
            q = self.q / (self.q.norm() + 1e-9)              # (d,)
            scores = (E @ q) / torch.clamp(self.tau, min=1e-2)
            scores = scores.masked_fill(mask == 0, -1e30)    # (B,S)
            w = torch.softmax(scores, dim=1)                 # (B,S)
            x = (E * w.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(self.mode)
        return F.layer_norm(x, (d,))  # 안정화
    


class VibTokeizerTrainer(L.LightningModule):
    def __init__(self, vib_tokenizer, llm, tokenizer, lr_head=1e-3, logit_scale_val=30.0):
        super().__init__()
        self.save_hyperparameters(ignore=["vib_tokenizer", "llm", "tokenizer"])
        self.vib_tokenizer = vib_tokenizer
        self.lr_head = lr_head

        d = int(llm.get_input_embeddings().embedding_dim)
        E = llm.get_input_embeddings()  # nn.Embedding(vocab, d)

        class_list = ['normal', 'unbalance', 'looseness', 'misalignment', 'bearing']
        emb_list = []
        for c in class_list:
            # add_special_tokens=False → 순수 서브워드만
            ids = tokenizer.encode(c, return_tensors='pt', add_special_tokens=False)  # (1, S)
            ids = ids.squeeze(0)  # (S,)
            # E.weight 인덱싱으로 안전하게 추출
            with torch.no_grad():
                tok_emb = E.weight[ids]           # (S, d)
                pooled = tok_emb.mean(dim=0)      # (d,)
                emb_list.append(pooled)

        W = torch.stack(emb_list, dim=0)          # (C, d), 원본 스케일 유지
        self.register_buffer("emb_matrix", W, persistent=False)
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale_val)), persistent=False)

    def training_step(self, batch, batch_idx):
        x = batch['x_stft']
        y = batch['x_cls']

        z = self.vib_tokenizer(x)        # (B, d)  
        t = self.emb_matrix[y]           # (B, d)  타깃 = LLM 원본 임베딩

        # 1) MSE: 좌표 그대로 맞추기
        loss_mse = F.mse_loss(z, t)

        # 2) Cosine: 방향(각도)도 맞추기
        loss_cos = 1.0 - F.cosine_similarity(z, t, dim=-1).mean()

        # 3) Norm match: 벡터 크기까지 맞추기 (LLM 분포 보정용)
        loss_norm = F.mse_loss(z.norm(dim=-1), t.norm(dim=-1))

        # 가중합 (경험상 아래 비율이 안정적)
        loss = 0.5*loss_mse + 0.4*loss_cos + 0.1*loss_norm

        # 로깅(배치 크기/분산 동기 포함)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log("train/mse", loss_mse, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        self.log("train/cos_loss", loss_cos, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        self.log("train/norm_loss", loss_norm, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))

        # 참고용: 현재 코사인 유사도/노름 통계
        with torch.no_grad():
            cos_sim = F.cosine_similarity(z, t, dim=-1).mean()
            self.log("train/cos_sim", cos_sim, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
            self.log("train/z_norm_mean", z.norm(dim=-1).mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
            self.log("train/t_norm_mean", t.norm(dim=-1).mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x_stft']
        y = batch['x_cls']
        z = self.vib_tokenizer(x)
        t = self.emb_matrix[y]

        loss_mse = F.mse_loss(z, t)
        loss_cos = 1.0 - F.cosine_similarity(z, t, dim=-1).mean()
        loss_norm = F.mse_loss(z.norm(dim=-1), t.norm(dim=-1))
        loss = 0.5*loss_mse + 0.4*loss_cos + 0.1*loss_norm

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log("val/mse", loss_mse, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        self.log("val/cos_loss", loss_cos, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        self.log("val/norm_loss", loss_norm, on_epoch=True, sync_dist=True, batch_size=x.size(0))

        with torch.no_grad():
            cos_sim = F.cosine_similarity(z, t, dim=-1).mean()
            self.log("val/cos_sim", cos_sim, on_epoch=True, sync_dist=True, batch_size=x.size(0))
            self.log("val/z_norm_mean", z.norm(dim=-1).mean(), on_epoch=True, sync_dist=True, batch_size=x.size(0))
            self.log("val/t_norm_mean", t.norm(dim=-1).mean(), on_epoch=True, sync_dist=True, batch_size=x.size(0))
        return loss

    def configure_optimizers(self):
        # alignment layer만 학습 (vib_tokenizer 내부 Encoder는 freeze 가정)
        opt = torch.optim.AdamW(self.vib_tokenizer.alignment_layer.parameters(),
                                lr=self.lr_head, weight_decay=1e-2)
        return opt