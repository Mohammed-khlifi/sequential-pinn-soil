import torch, torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # your base MLP: ℝ²→ℝ¹
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Q, K, V all map ℝ¹→ℝ¹²⁸
        self.Q = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 128)
        )
        self.K = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 128)
        )
        self.V = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 128)
        )
        # project the 128‐dim attention output back to 1
        self.proj = nn.Linear(128, 1)

    def forward(self, zt):
        # zt: [batch, 2]
        h_1 = self.net(zt)         # [batch, 1]
        Q = self.Q(h_1)           # [batch, 128]
        K = self.K(h_1)           # [batch, 128]
        V = self.V(h_1)           # [batch, 128]

        d_k = Q.size(-1)          # 128
        # compute scaled dot‐product attention across the batch dimension
        scores = Q*K / math.sqrt(d_k)  # [batch, batch]
        A = torch.softmax(scores, dim=1)                            # [batch, batch]
   
        attn = A*V                                   # [batch, 128]

        # bring it back down to [batch,1] so we can add to h_1
        attn = self.proj(attn)                                      # [batch, 1]

        return h_1 + attn

