import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

class PatchEmbed(nn.Module):
    """
    Splits the spectrogram into patches and projects each patch to an embedding.
    """
    def __init__(self, 
                 in_channels: int, 
                 patch_size: tuple, 
                 emb_dim: int):
        """
        Args:
            in_channels: Number of input channels (e.g. 1 for a single spectrogram channel).
            patch_size: (time_patch, freq_patch) sizes for splitting the spectrogram.
            emb_dim: Dimension of the linear projection.
        """
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        # Each patch is shape (in_channels, patch_time, patch_freq). We'll flatten and project it.
        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1], emb_dim)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, T, F) — a 4D tensor of batched spectrograms
        Returns:
            patch_embeddings: (B, num_patches, emb_dim)
        """
        B, C, T, F = x.shape
        pt, pf = self.patch_size

        # Check divisibility to avoid shape mismatch
        assert T % pt == 0, f"Time dimension must be divisible by patch_time. Got T={T}, patch_time={pt}."
        assert F % pf == 0, f"Freq dimension must be divisible by patch_freq. Got F={F}, patch_freq={pf}."

        # Unfold into patches: shape => (B, C, T//pt, F//pf, pt, pf)
        patches = x.unfold(dimension=2, size=pt, step=pt).unfold(dimension=3, size=pf, step=pf)
        # Flatten out the patch dimensions
        # Current shape: (B, C, #patches_time, #patches_freq, pt, pf)
        patches = patches.contiguous().view(B, C, -1, pt, pf)  # => (B, C, Nx, pt, pf), Nx = total patches
        # Rearrange to (B, Nx, C, pt, pf)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        # Finally flatten each patch into a vector
        patches = patches.view(B, -1, C * pt * pf)
        
        # Linear projection
        patch_embeddings = self.proj(patches)  # (B, Nx, emb_dim)
        return patch_embeddings


class AudioTransformer(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_mels=64,
                 n_fft=1024,
                 hop_length=512,
                 patch_time=6,
                 patch_freq=16,
                 emb_dim=128,
                 num_heads=4,
                 num_layers=4,
                 num_classes=10,
                 hidden_dim=256,
                 dropout=0.1):
        """
        Transformer for 1s audio classification. 
        Adjusted so time and freq dimensions are divisible by patch sizes.
        """
        super().__init__()

        self.mel_spec_transform = nn.Sequential(
            MelSpectrogram(
                sample_rate=sample_rate, 
                n_mels=n_mels, 
                n_fft=n_fft, 
                hop_length=hop_length
            ),
            AmplitudeToDB()
        )

        self.patch_size = (patch_time, patch_freq)

        self.patch_embed = PatchEmbed(
            in_channels=1,
            patch_size=self.patch_size,
            emb_dim=emb_dim
        )

        # Number of patches = (T/patch_time) * (F/patch_freq)
        # +1 for the [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, 500, emb_dim))  # can set bigger if needed

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        for name, param in self.named_parameters():
            if param.dim() > 1 and 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, waveforms):
        """
        Args:
            waveforms: (B, 1, num_samples) at 16kHz (1 second => 16000 samples).
        Returns:
            logits: (B, num_classes)
        """
        # waveform -> MelSpectrogram => (B, n_mels, T)
        mel_specs = self.mel_spec_transform(waveforms)  
        # => (B, n_mels, T)

        # Reshape to (B, 1, T, n_mels) for patching
        mel_specs = mel_specs.unsqueeze(1).transpose(2, 3)
        # => shape (B, 1, T, n_mels)

        x = self.patch_embed(mel_specs)  # (B, num_patches, emb_dim)

        B, N, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # => (B, N+1, emb_dim)

        # positional embeddings
        x = x + self.pos_embed[:, :N+1, :]

        x = self.transformer_encoder(x)  # => (B, N+1, emb_dim)

        # classification on CLS token
        cls_rep = x[:, 0]  # (B, emb_dim)
        logits = self.mlp_head(cls_rep)  # (B, num_classes)
        return logits