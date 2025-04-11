import torch
import torch.nn as nn
import numpy as np
import copy

from itertools import combinations

class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, output_size = 256, filter_size = 5, num_blocks = 12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(in_channel, 32, 3, 2, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class EncoderSplit(Encoder):
    def __init__(self, num_epi, output_size = 256, filter_size = 5, num_blocks = 12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        self.conv_start_epi = nn.Sequential(
                                    nn.Conv1d(num_epi, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):

        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        epi = self.res_blocks_epi(self.conv_start_epi(epi))

        x = torch.cat([seq, epi], dim = 1)
        out = self.conv_end(x)
        return out
    
class EncoderCrossAttn(Encoder):
    def __init__(self, num_epi_layers, output_size=256, filter_size=5, num_blocks=12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size

        # Sequence layer encoder
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )

        # Epi layer encoder
        self.conv_epi_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, 3, 2, 1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
            ) for i in range(num_epi_layers)
        ])

        hiddens =        [32, 32, 32, 64, 64, 128, 128, 128, 256, 256, 512, 512]
        hidden_ins = [32, 32, 32, 32, 64, 64, 128, 128, 128, 256, 256, 512]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)

        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(hidden_ins_half[-1], 2, dropout=0.1, batch_first=True)
            #TransformerLayer(hiddens[-1], nhead=4, dropout=0.1, dim_feedforward=512, batch_first=True)
            for i in range(num_epi_layers ** 2)
        ])

        # Residual blocks
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_epi = nn.ModuleList([
            self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half) for i in range(num_epi_layers)
        ])

        self.conv_end = nn.Conv1d(12800, output_size, 1)

    def forward(self, x):
        x_seq = x[:, :5, :]
        x_epi = x[:, 5:, :]

        # Epi layer encoder
        x_epi_list = [self.conv_epi_layers[i](x_epi[:, i:i + 1, :]) for i in range(len(self.res_blocks_epi))]

        x_seq = self.res_blocks_seq(self.conv_start_seq(x_seq))
        for j in range(len(self.res_blocks_epi)):
            x_epi_list[j] = self.res_blocks_epi[j](x_epi_list[j])

        # Cross-attention blocks
        cross_attn_out = []
        cross_attn_mats = []
        #for block_i, (i, j) in enumerate(combinations(range(len(x_epi_list)), 2)):
        for i in range(len(x_epi_list)):
            for j in range(len(x_epi_list)):
                block_i = i * len(x_epi_list) + j
                cross_attn, attn_weights = self.cross_attn_blocks[block_i](x_epi_list[i], x_epi_list[j], x_epi_list[i], need_weights=True, average_attn_weights=False)
                cross_attn_out.append(cross_attn)
                cross_attn_mats.append([cross_attn, attn_weights])
        # TODO: test avg/max pooling of cross attn to allow for dynamic number of epi layers

        # for i in range(len(self.cross_attn_blocks)):
        #     cross_attn, attn_weights = self.cross_attn_blocks[i](x_epi_list[i + 1], x_epi_list[i], x_epi_list[i])
        #     cross_attn_out.append(cross_attn)

        # Concatenate sequence and epi layer encodings
        cross_attn_max = torch.cat(cross_attn_out, dim=1)
        x = torch.cat([x_seq, cross_attn_max], dim=1)
        # Final linear layer
        x = self.conv_end(x)

        return x, cross_attn_mats


class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil 
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out

class Decoder2D(nn.Module):
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 5):
        super(Decoder2D, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    
# --- NEW: 1D Decoder ---
class Decoder1D(nn.Module):
    """
    Decodes latent representation back to 1D tracks.
    Uses Transposed Convolutions for upsampling.
    """
    def __init__(self, num_target_tracks, latent_dim=256, latent_length=None, target_length=2097152, num_upsample_blocks=7):
        super(Decoder1D, self).__init__()
        self.num_target_tracks = num_target_tracks
        self.target_length = target_length
        self.latent_dim = latent_dim # Channels in the latent space
        # num_upsample_blocks should match the number of downsampling steps in the encoder
        # Example: If Encoder reduces length by 2^13, we need 13 upsampling steps of factor 2.

        # Define upsampling blocks
        blocks = []
        current_dim = latent_dim
        # Hidden dimensions for upsampling layers (can be designed, e.g., decreasing)
        # Simple approach: halve dimension at each step until a minimum?
        # Let's use constant hidden dim for simplicity first
        hidden_dim = 128 # Or relate it to latent_dim

        # Initial Conv to potentially adjust channels before upsampling
        blocks.append(nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ))
        current_dim = hidden_dim

        # Upsampling using ConvTranspose1d
        for i in range(num_upsample_blocks):
             out_dim = hidden_dim # Keep hidden dim constant for now
             blocks.append(nn.Sequential(
                 # Stride=2 doubles the length
                 # Kernel size affects overlap, padding affects output size alignment
                 # Careful calculation needed for kernel/padding/output_padding
                 # Kernel=4, Stride=2, Padding=1 generally works for doubling length
                 nn.ConvTranspose1d(current_dim, out_dim, kernel_size=4, stride=2, padding=1),
                 nn.BatchNorm1d(out_dim),
                 nn.ReLU()
             ))
             current_dim = out_dim

        self.upsample_blocks = nn.Sequential(*blocks)

        # Final convolution to match the target number of tracks and potentially adjust final length slightly if needed
        # Calculate expected length after upsampling
        # Need latent_length if not None. Assume it's input_len / (2^num_upsample_blocks)
        # expected_len = latent_length * (2**num_upsample_blocks) if latent_length else None
        # This final layer adjusts channels. We might need a final conv/padding if length isn't exact.
        self.final_conv = nn.Conv1d(current_dim, num_target_tracks, kernel_size=1)

        # Optional: Add padding/cropping layer if upsampling doesn't hit target_length exactly
        # self.final_adjust = nn.AdaptiveAvgPool1d(target_length) # Or padding

    def forward(self, x):
        # Input x shape: [batch, latent_dim, latent_length]
        x = self.upsample_blocks(x)
        # Output shape: [batch, hidden_dim, upsampled_length]
        x = self.final_conv(x)
        # Output shape: [batch, num_target_tracks, upsampled_length]

        # Adjust length if needed (e.g., crop or pad)
        current_length = x.shape[-1]
        if current_length != self.target_length:
            # Simple padding/cropping (center crop/pad if possible)
            if current_length > self.target_length:
                diff = current_length - self.target_length
                start = diff // 2
                x = x[..., start : start + self.target_length]
            else:
                diff = self.target_length - current_length
                pad_start = diff // 2
                pad_end = diff - pad_start
                x = nn.functional.pad(x, (pad_start, pad_end))

        # Output shape: [batch, num_target_tracks, target_length]
        # change to [batch, target_length, num_target_tracks] if needed
        x = x.permute(0, 2, 1)
        return x

class TransformerLayer(torch.nn.TransformerEncoderLayer):
    # Pre-LN structure
    
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        # MHA section
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm, 
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)

        # MLP section
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights

class TransformerEncoder(torch.nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None, record_attn = False):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        attn_weight_list = []

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
        if self.norm is not None:
            output = self.norm(output)

        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        else:
            return output

    def _get_clones(self, module, N):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):

    def __init__(self, hidden, dropout = 0.1, max_len = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttnModule(nn.Module):
    def __init__(self, hidden = 128, layers = 8, record_attn = False, inpu_dim = 256):
        super(AttnModule, self).__init__()

        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout = 0.1)
        encoder_layers = TransformerLayer(hidden, 
                                          nhead = 8,
                                          dropout = 0.1,
                                          dim_feedforward = 512,
                                          batch_first = True)
        self.module = TransformerEncoder(encoder_layers, 
                                         layers, 
                                         record_attn = record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.module(x)
        return output

    def inference(self, x):
        return self.module(x)

if __name__ == '__main__':
    main()
