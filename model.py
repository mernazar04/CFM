
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),

            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )

        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ConditionalFlowModel(nn.Module):
    def __init__(self, in_channels=16, hidden=32, num_classes=2):
        super().__init__()

        self.class_embedding = nn.Embedding(num_classes + 1, hidden)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        self.input_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)

        self.res_blocks = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden)
        )

        self.output_conv = nn.Conv2d(hidden, in_channels, 3, padding=1)

    def forward(self, x, t, c):
        # x: (B, 16, 128, 128)
        # t: (B,)
        # c: (B,)

        h = self.input_conv(x)

        time_emb = self.time_mlp(t.unsqueeze(1))
        class_emb = self.class_embedding(c)

        condition = time_emb + class_emb

        h = h + condition[:, :, None, None]

        h = self.res_blocks(h)

        return self.output_conv(h)


if __name__ == "__main__":
    model = ConditionalFlowModel()

    x = torch.randn(4, 16, 128, 128)
    t = torch.rand(4)
    c = torch.tensor([1, 2, 1, 2])

    output = model(x, t, c)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))