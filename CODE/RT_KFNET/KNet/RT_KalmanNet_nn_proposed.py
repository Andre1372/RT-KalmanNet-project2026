# -*- coding: utf-8 -*-
"""In this file we build the neural network that estimates the robustness
parameter c of the REKF.
It is composed by
1) a fully connected encoder (Linear -> LayerNorm -> GELU -> Linear -> GELU)
2) a single GRU layer, whose hidden state carries the recurrent memory
3) a fully connected output layer followed by a sigmoid
"""

import torch
import torch.nn as nn


class RT_KalmanNet_nn(nn.Module):
    """Estimates the REKF robustness parameter c from a recurrent encoder-GRU-decoder network."""

    def __init__(
        self, 
        input_size_fcl: int, 
        gru_hidden_size: int = 64, 
        gru_layers: int = 1, 
        dropout: float = 0.0,
    ) -> None:
        """Initializes the encoder, GRU, and output layers.

        Args:
            input_size_fcl: Number of input neurons to the encoder.
            gru_hidden_size: Size of the GRU hidden state (= size of the encoder output).
            gru_layers: Number of GRU layers.
            dropout: Dropout between GRU layers (ignored if gru_layers == 1).
        """
        super().__init__()

        self.gru_hidden_size = gru_hidden_size
        self.gru_layers = gru_layers

        # Fully connected encoder
        self.enc = nn.Sequential(
            nn.Linear(input_size_fcl, gru_hidden_size),
            nn.LayerNorm(gru_hidden_size),
            nn.GELU(),
            nn.Linear(gru_hidden_size, gru_hidden_size),
            nn.GELU(),
        )

        # Single GRU block: replaces the hand-written feedback loop.
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            dropout=(dropout if gru_layers > 1 else 0.0),
            batch_first=False,
        )

        # Fully connected output layer
        self.output_layer = nn.Linear(gru_hidden_size, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        # Zero bias so that c starts at sigmoid(0) = 0.5, the same starting point as
        # the Original network: the comparison between the two architectures is then
        # not biased by the initialization.
        nn.init.constant_(self.output_layer.bias, 0.0)

    def init_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Builds a zero-initialized hidden state, shape [gru_layers, batch_size, gru_hidden_size].

        Must be called explicitly by the caller at the start of a sequence
        (with the returned h_t then passed back into forward() at every
        subsequent step).
        """
        return torch.zeros(self.gru_layers, batch_size, self.gru_hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs one timestep through the encoder, GRU, and output layer.

        Args:
            x: Network input (row vector of size input_size_fcl, relative to
                a single timestep).
            h_t: GRU hidden state from the previous step, shape
                [gru_layers, batch_size, gru_hidden_size]. 
        Returns:
            Tuple of:
                c_t: Scalar in (0, 1), the estimated robustness parameter.
                h_new: New GRU hidden state, to be passed back into the next
                    call to forward().
        """
        z = self.enc(x)

        # nn.GRU (batch_first=False) expects input of shape [seq_len, batch, features].
        z = z.view(1, 1, -1)

        # h_t is never detached here: the gradient can flow backward through
        # time via h_t until the caller explicitly truncates it (h_t.detach())
        # for truncated BPTT.
        out, h_new = self.gru(z, h_t)

        raw = self.output_layer(out.view(1, -1))
        c_t = torch.sigmoid(raw).squeeze()

        return c_t, h_new