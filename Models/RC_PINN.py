import torch, torch.nn as nn
import torch.nn.functional as F


class RC_PINN(nn.Module):
    def __init__(self, 
                 conv_layers=[(3, 8, 3)],  # List of (in_channels, out_channels, kernel_size)
                 linear_layers=[(8, 64), (64, 64), (64, 1)],
                 activation=nn.ReLU()):
        super().__init__()

        # Build CNN layers
        conv_modules = []
        for i, (in_ch, out_ch, ksize) in enumerate(conv_layers):
            conv_modules.append(nn.Conv1d(in_ch, out_ch, kernel_size=ksize, padding='same'))
            if i < len(conv_layers) - 1:
                conv_modules.append(activation)
        self.conv_net = nn.Sequential(*conv_modules)

        # Last CNN layer output channels (used as input to FC layers)
        final_out_ch = conv_layers[-1][1]

        # Build the fully connected layers
        adjusted_linear_layers = [(final_out_ch if i == 0 else in_dim, out_dim)
                                  for i, (in_dim, out_dim) in enumerate(linear_layers)]

        fc_modules = []
        for i, (in_dim, out_dim) in enumerate(adjusted_linear_layers):
            fc_modules.append(nn.Linear(in_dim, out_dim))
            if i < len(adjusted_linear_layers) - 1:
                fc_modules.append(activation)
        self.fc_net = nn.Sequential(*fc_modules)
        
        self.net2 = nn.Sequential(
            nn.Linear(1,  64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64,  1),
        )

    def forward(self, t , z , h):
        input_features = torch.cat([t.unsqueeze(-1) , z.unsqueeze(-1) , h] ,  dim = -1).unsqueeze(1)
        # Input: [batch, time, features] → [batch, features, time]
        x = input_features.permute(0, 2, 1)

        # Apply CNN layers
        x = self.conv_net(x)  # shape: [batch, out_ch, time]

        # Convert back to [batch, time, channels] for FC layers
        x = x.permute(0, 2, 1)
        h_mlp = self.net2(input_features[: , : , -1].unsqueeze(0))
        # Apply FC layers (per time step)
        out = self.fc_net(x).squeeze() + input_features[: , : , -1].squeeze() # (h_mlp * input_features[: , : , -1]).squeeze()# shape: [batch, time, output_dim]
        

        return out

