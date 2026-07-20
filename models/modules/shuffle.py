import torch
import torch.nn as nn
import torch.nn.functional as F

def masking_generator(x):
    _, H, W, _ = x.shape
    threshold = 16
    mask_value_far = 1.
    mask_value_close = 0.
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).reshape(-1, 2).float()
    dist_sq = torch.cdist(coords, coords, p=2).pow(2)  # (HW, HW)
    dist = dist_sq.sqrt()

    # 거리 기준에 따라 마스크 생성
    mask = torch.where(dist > threshold, mask_value_far, mask_value_close)
    return mask

class InvertGaussianEncoder(nn.Module):
    def __init__(self, H, W, spread=10.0):
        super().__init__()
        self.H = H
        self.W = W
        self.spread = spread
        HW = H * W

        self.attn_logits = nn.Parameter(torch.randn(HW, HW))  # 학습 대상

        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).reshape(-1, 2).float()
        dist_sq = torch.cdist(coords, coords, p=2).pow(2)  # (HW, HW)
        self.register_buffer('dist_sq', dist_sq)

    def forward(self, x):
        B, C, H, W = x.shape
        print(f"Encoder input shape: {x.shape}")
        HW = H * W
        x_flat = x.view(B, C, HW)
        attn = F.softmax(self.attn_logits, dim=1)
        x_attn = torch.matmul(x_flat, attn)
        x_out = x_attn.view(B, C, H, W)
        return x_out, attn

    def distance_loss(self):
        attn = F.softmax(self.attn_logits, dim=1)
        log_attn = F.log_softmax(self.attn_logits, dim=1)

        # 가까운 데 attention 주면 penalty
        penalty_weight = torch.exp(-self.dist_sq / (2 * self.spread**2))
        penalty_weight = penalty_weight / penalty_weight.sum(dim=1, keepdim=True)
        penalty_loss = (attn * penalty_weight).sum(dim=1).mean()

        # attention이 퍼지면 손해
        entropy_penalty = - (attn * log_attn).sum(dim=1).mean()

        return penalty_loss + 2.0 * entropy_penalty, penalty_weight
    
    def compute_penalty_weight(self, dist_sq, spread=5.0, mode='gauss'):
        if mode == 'gauss':
            weight = torch.exp(-dist_sq / (2 * spread**2))  # Gaussian
        elif mode == 'sqrt':
            weight = torch.exp(-torch.sqrt(dist_sq) / spread)  # smoother
        else:
            raise ValueError("지원되는 모드는 'gauss' 또는 'sqrt'만 가능")
        return weight / weight.sum(dim=1, keepdim=True)  # normalize


    
class InvertGaussianDecoder(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        HW = H * W

        self.attn_logits = nn.Parameter(torch.randn(HW, HW))

    def forward(self, x_encoded):
        """
        x_encoded: (B, 3, H, W)
        returns: (B, 3, H, W), (HW, HW) = x_recon, decoder_attn
        """
        B, C, H, W = x_encoded.shape
        HW = H * W
        x_flat = x_encoded.view(B, C, HW)
        attn = self.attn_logits
        x_recon = torch.matmul(x_flat, attn)       # (B, C, HW)
        x_out = x_recon.view(B, C, H, W)
        return x_out, attn  # return attention matrix



# === 1. ViT Block === #
class ViTBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights  # attn_weights: (B, N, N)

# === 2. EncoderViT === #
class EncoderViT(nn.Module):
    def __init__(self, H, W, patch_size=4, embed_dim=96, depth=4, heads=4):
        super().__init__()
        self.H, self.W = H, W
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.N = self.Hp * self.Wp
        self.HW = H * W

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.transformer_blocks = nn.ModuleList([
            ViTBlock(embed_dim, heads=heads) for _ in range(depth)
        ])

        self.upsample_cnn = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_patch = self.patch_embed(x)  # (B, D, Hp, Wp)
        x_seq = x_patch.flatten(2).transpose(1, 2)  # (B, N, D)

        attn_last = None
        for block in self.transformer_blocks:
            x_seq, attn_last = block(x_seq)

        attn_img = attn_last.unsqueeze(1)  # (B, 1, N, N)
        attn_full = self.upsample_cnn(attn_img)  # (B, HW, HW)

        x_flat = x.view(B, C, -1)  # (B, C, HW)
        
        x_attn = torch.matmul(x_flat, attn_full)  # (B, C, HW)
        x_out = x_attn.view(B, C, H, W)
        return x_out, attn_full


# === 3. DecoderViT === #
class DecoderViT(nn.Module):
    def __init__(self, H, W, patch_size=4, embed_dim=96, depth=4, heads=4):
        super().__init__()
        self.H, self.W = H, W
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.N = self.Hp * self.Wp
        self.HW = H * W

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.transformer_blocks = nn.ModuleList([
            ViTBlock(embed_dim, heads=heads) for _ in range(depth)
        ])

        self.upsample_cnn = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x_encoded):
        B, C, H, W = x_encoded.shape
        HW = H * W

        x_patch = self.patch_embed(x_encoded)                # (B, D, Hp, Wp)
        x_seq = x_patch.flatten(2).transpose(1, 2)           # (B, N, D)

        attn_last = None
        for block in self.transformer_blocks:
            x_seq, attn_last = block(x_seq)                  # Extract attention from last block

        attn_img = attn_last.unsqueeze(1)                    # (B, 1, N, N)
        decoder_attn = self.upsample_cnn(attn_img).squeeze(1)  # (B, HW, HW)

        x_flat = x_encoded.view(B, C, HW)                    # (B, C, HW)
        x_recon = torch.matmul(x_flat, decoder_attn)         # (B, C, HW)
        x_recon = x_recon.view(B, C, H, W)                   # (B, C, H, W)

        return x_recon, decoder_attn  # Return both reconstructed image and attention matrix


# === 4. Utility Functions === #

def apply_mask(x_encoded, x_other, mask):
    """
    x_encoded: (B, 3, H, W)
    x_other: (B, 3, H, W)
    mask: (B, 1, H, W) where 1=keep, 0=replace
    """
    return x_encoded * mask + x_other * (1 - mask)

def identity_loss(decoder_attn, encoder_attn):
    """
    decoder_attn, encoder_attn: (B, HW, HW)
    목표: decoder_attn @ encoder_attn ≈ Identity
    """
    B, HW, _ = decoder_attn.shape
    I = torch.eye(HW, device=decoder_attn.device).unsqueeze(0).expand(B, -1, -1)  # (B, HW, HW)
    prod = torch.matmul(decoder_attn, encoder_attn)  # (B, HW, HW)
    loss = F.mse_loss(prod, I)
    return loss
