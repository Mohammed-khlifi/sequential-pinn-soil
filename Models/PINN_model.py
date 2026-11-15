import torch
import torch.nn as nn
from typing import Union, Sequence

def get_activation(act: Union[str, nn.Module]) -> nn.Module:
    """Return an activation module from a string or pass-through a given module."""
    if isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act in ("tanh",):
        return nn.Tanh()
    if act in ("relu",):
        return nn.ReLU()
    if act in ("gelu",):
        return nn.GELU()
    if act in ("silu", "swish"):
        return nn.SiLU()
    if act in ("elu",):
        return nn.ELU()
    if act in ("leakyrelu", "lrelu"):
        return nn.LeakyReLU(0.01)
    if act in ("sin", "sine"):
        class SineActivation(nn.Module):
            def forward(self, x):
                return torch.sin(x)
        return SineActivation()
    if act in ("cos", "cosine"):
        class CosineActivation(nn.Module):
            def forward(self, x):
                return torch.cos(x)
        return CosineActivation()
    if act in ("softplus",):
        return nn.Softplus()
    if act in ("sigmoid",):
        return nn.Sigmoid()
    if act in ("adaptive",):
        class AdaptiveTanh(nn.Module):
            def __init__(self, init_alpha=1.0):
                super().__init__()
                self.alpha = nn.Parameter(torch.tensor(init_alpha))

            def forward(self, x):
                return torch.tanh(self.alpha * x)
        return AdaptiveTanh()

    raise ValueError(f"Unknown activation: {act}")

class PINN(nn.Module):
    """
    Configurable Physics-Informed Neural Network (fully-connected MLP).

    Args:
        input_dim:   number of input features (e.g., 2 for [z, t])
        output_dim:  number of outputs (e.g., 1 for head h)
        hidden_dim:  width of each hidden layer
        num_hidden:  number of hidden layers
        activation:  activation name ('tanh','relu','gelu','silu',...) or nn.Module
        layer_sizes: optional explicit list of hidden sizes (overrides hidden_dim/num_hidden)
                     e.g., [128, 128, 64] -> three hidden layers with those widths
    """
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_hidden: int = 2,
        activation: Union[str, nn.Module] = "tanh",
        layer_sizes: Sequence[int] = None,
    ):
        super().__init__()
        act = get_activation(activation)

        if layer_sizes is not None and len(layer_sizes) > 0:
            sizes = [input_dim] + list(layer_sizes) + [output_dim]
        else:
            sizes = [input_dim] + [hidden_dim] * num_hidden + [output_dim]

        layers = []
        for in_f, out_f in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_f, out_f))
            # add activation after every layer except the last
            if out_f != output_dim:
                layers.append(act.__class__() if isinstance(act, nn.Module) else act)

        self.net = nn.Sequential(*layers)

    def forward(self, Z: torch.Tensor = None, T: torch.Tensor = None, X: torch.Tensor = None):
        """
        Forward call:
        - Either pass Z and T (1D or any broadcastable shapes) and we'll concat as [*, 2].
        - Or pass X directly with shape [batch, input_dim].
        """
        if X is None:
            if Z is None or T is None:
                raise ValueError("Provide either (Z and T) or X.")
            X = self.make_input(Z, T)
        return self.net(X)

    @staticmethod
    def make_input(Z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Concat Z and T along the last dim -> shape [..., 2]."""
        if Z.ndim == T.ndim:
            return torch.cat([Z.unsqueeze(-1), T.unsqueeze(-1)], dim=-1)
        # try to broadcast if shapes differ (e.g., [N,] and [M,])
        Zb, Tb = torch.broadcast_tensors(Z, T)
        return torch.cat([Zb.unsqueeze(-1), Tb.unsqueeze(-1)], dim=-1)
