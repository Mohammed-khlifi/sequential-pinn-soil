
import torch
import torch.nn as nn




class DSConvBlock(nn.Module):
    def __init__(self, channels, expansion=2, dropout=0.0, groups=8):
        super().__init__()
        hidden = channels * expansion

        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1,  padding_mode='replicate', groups=channels, bias=False)
        
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.ReLU()
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

        self.norm1 = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 4), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(max(channels // 8, 4), channels, kernel_size=1),
            nn.Sigmoid(),
        )



    def forward(self, x):
        r = x

        x = self.dw(x)
        x = self.act(x)
        x = nn.ReLU()(x)
        x = self.pw1(x)
        x = nn.ReLU()(x)
        x = self.drop(x)
        x = self.pw2(x)
        x += r

        x = self.norm2(x)
        x = x * self.se(x)
        return x


class DSSE_Net(nn.Module):
    """
    Drop-in replacement for FastPDEBlock:
      - richer features (residual DSConv blocks + SE attention)
      - faster than standard heavy conv stacks (depthwise separable convs)
      - stable updates via learnable residual step-size
    """
    def __init__(self, in_channels=3, width=64, depth=6, dropout=0.05, groups=8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, width), num_channels=width),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(*[
            DSConvBlock(width, expansion=2, dropout=dropout, groups=groups)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, width // 2), num_channels=width // 2),
            nn.ReLU(),
            nn.Conv2d(width // 2, 1, kernel_size=1, bias=False),
        )

        

        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_input(self, z, x, h_old):
        if z.dim() == 2:
            z = z.unsqueeze(0)
            x = x.unsqueeze(0)
            h_old = h_old.unsqueeze(0)

        # (B, Nz, Nx, 3) -> (B, 3, Nz, Nx)
        inp = torch.stack([z, x, h_old], dim=-1).permute(0, 3, 1, 2).contiguous()
        #print(inp.shape, h_old.unsqueeze(-1).permute(0, 3, 1, 2).contiguous().shape)
        return inp #h_old.unsqueeze(-1).permute(0, 3, 1, 2).contiguous()

    def forward(self, z, x, h_old):
        squeeze_back = (h_old.dim() == 2)

        inp = self._make_input(z, x, h_old)
        feat = self.stem(inp)
        deep = self.blocks(feat)
        combined = feat + deep
        delta_h = self.head(combined).squeeze(1)  # (B, Nz, Nx)

        h_new = h_old + delta_h
        if squeeze_back:
            h_new = h_new.squeeze(0)
        return h_new