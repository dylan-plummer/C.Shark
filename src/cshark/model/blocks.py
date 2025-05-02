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
        # compute block num where we reach min_size at which point the stride is 1
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class EncoderSplit(Encoder):
    def __init__(self, num_epi, hidden = 256, output_size = 256, filter_size = 5, num_blocks = 12, num_bases=5):
        super(Encoder, self).__init__()
        self.num_epi = num_epi
        self.filter_size = filter_size
        self.num_bases = num_bases
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(num_bases, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        if num_epi > 0:
            self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(num_epi, 16, 3, 2, 1),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        if num_epi == 0:
            hiddens_half[-1] *= 2
            if num_blocks == 11:
                hiddens_half[-2] *= 2
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        if num_epi > 0:
            self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.conv_end = nn.Conv1d(256, hidden, 1)

    def forward(self, x):
        if self.num_epi > 0:
            seq = x[:, :self.num_bases, :]
            epi = x[:, self.num_bases:, :]
            seq = self.res_blocks_seq(self.conv_start_seq(seq))
            epi = self.res_blocks_epi(self.conv_start_epi(epi))
            x = torch.cat([seq, epi], dim = 1)
        else:
            x = self.res_blocks_seq(self.conv_start_seq(x))
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


class ResBlockDilated1D(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated1D, self).__init__()
        pad_len = dil 
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm1d(hidden),
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
    def __init__(self, num_target_tracks, latent_dim=256, filter_size=3, num_blocks=7, 
                 target_length=2097152, num_upsample_blocks=3):
        super(Decoder1D, self).__init__()
        self.num_target_tracks = num_target_tracks
        self.target_length = target_length
        self.latent_dim = latent_dim # Channels in the latent space
        self.filter_size = filter_size
        self.num_blocks = num_blocks

        self.conv_start = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=2, stride=2),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=filter_size, padding='same'),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
            ) for _ in range(num_upsample_blocks)
        ])
        self.res_blocks = self.get_res_blocks(num_blocks, latent_dim)
        self.conv_end = nn.Conv1d(latent_dim, num_target_tracks, kernel_size=1)

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated1D(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

    def forward(self, x):
        x = self.conv_start(x)
        for block in self.upsample_blocks:
            x = block(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        out = out.permute(0, 2, 1)
        return out

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
