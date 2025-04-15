import math
import torch
import torch.nn as nn

import cshark.model.blocks as blocks

class MultiTaskConvTransModel(nn.Module): # Renamed for clarity
    """
    Predicts both 2D Hi-C maps and 1D tracks.
    """
    def __init__(self, num_genomic_features,
                 num_target_tracks=0, # Number of 1D tracks to predict
                 mid_hidden = 256,    # Latent dimension size
                 predict_hic = True,  # Whether to include the Hi-C prediction head
                 predict_1d = False,  # Whether to include the 1D prediction head
                 target_1d_length = 2048, # Expected output length for 1D tracks
                 encoder_downsample_factor = 2**7, # Total downsampling from encoder (e.g., 13 blocks * stride 2)
                 record_attn = False):
        super(MultiTaskConvTransModel, self).__init__()

        if not predict_hic and not predict_1d:
            raise ValueError("Model must be configured to predict at least Hi-C or 1D tracks.")
        if predict_1d and num_target_tracks <= 0:
            raise ValueError("If predict_1d is True, num_target_tracks must be > 0.")

        print(f'Initializing MultiTaskConvTransModel:')
        print(f'  Predicting Hi-C: {predict_hic}')
        print(f'  Predicting 1D Tracks: {predict_1d}')
        if predict_1d:
            print(f'  Number of target 1D tracks: {num_target_tracks}')

        self.predict_hic = predict_hic
        self.predict_1d = predict_1d
        self.num_target_tracks = num_target_tracks
        self.record_attn = record_attn
        self.encoder_downsample_factor = encoder_downsample_factor
        self.target_1d_length = target_1d_length

        # --- Encoder ---
        # Takes sequence (5) + genomic features
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        # Output: [batch, mid_hidden, reduced_length]

        # --- Optional Transformer ---
        # Operates on the latent sequence
        self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn)
        # Output: [batch, reduced_length, mid_hidden] if AttnModule uses batch_first, or tuple if record_attn

        # --- Decoders ---
        # 2D Decoder for Hi-C
        if self.predict_hic:
             # Input channels based on diagonalize logic (mid_hidden * 2)
            self.decoder_2d = blocks.Decoder2D(mid_hidden * 2) # Use renamed Decoder2D
            # Output: [batch, 1, hic_map_size, hic_map_size]

        # 1D Decoder for Tracks
        if self.predict_1d:
            print(f"1D Decoder using latent dim={mid_hidden}{target_1d_length}")

            self.decoder_1d = blocks.Decoder1D(num_target_tracks = self.num_target_tracks,
                                               latent_dim=mid_hidden,
                                               target_length=self.target_1d_length)
            # Output: [batch, num_target_tracks, target_1d_length]

    def forward(self, x):
        '''
        Input feature x: [batch_size, length, feature_dim (5 + num_genomic_features)]
        '''
        # 1. Permute to [batch, features, length] for Conv1d
        x = self.move_feature_forward(x).float()
        # Shape: [batch, 5 + num_genomic_features, length]

        # 2. Encode
        latent_seq = self.encoder(x)
        # Shape: [batch, mid_hidden, reduced_length]

        # 3. Optional Transformer (Attention)
        # Needs input as [batch, seq_len, features] if batch_first=True in AttnModule
        latent_seq_permuted = self.move_feature_forward(latent_seq)
        # Shape: [batch, reduced_length, mid_hidden]

        attn_weights = None
        if self.record_attn:
            attn_output = self.attn(latent_seq_permuted) # Expects tuple (output, weights)
            if isinstance(attn_output, tuple):
                 latent_transformed = attn_output[0]
                 attn_weights = attn_output[1]
            else: # Should not happen if record_attn is True and AttnModule is correct
                 latent_transformed = attn_output
                 print("Warning: record_attn=True but AttnModule did not return weights.")
        else:
            attn_output = self.attn(latent_seq_permuted) # Expects tensor output
            if isinstance(attn_output, tuple): # If attn module *always* returns tuple
                 latent_transformed = attn_output[0]
            else:
                 latent_transformed = attn_output
        # Shape: [batch, reduced_length, mid_hidden]

        # Permute back to [batch, mid_hidden, reduced_length] for decoders
        latent_final = self.move_feature_forward(latent_transformed)
        # Shape: [batch, mid_hidden, reduced_length]

        # 4. Decode
        outputs = {}
        if self.predict_hic:
            # Diagonalize latent features for 2D decoder
            diag_input = self.diagonalize(latent_final)
            # Shape: [batch, mid_hidden * 2, reduced_length, reduced_length] (approx)
            # The size of the diagonalized map depends on reduced_length, needs to match decoder input expectation (e.g., 256x256)
            # This implies reduced_length should be ~256. Check encoder design.
            # If reduced_length is not 256, diagonalize/decoder needs adjustment or interpolation.
            # Assuming reduced_length matches the Hi-C map size expected by Decoder2D (e.g., 256 after resize)
            # Let's assume diagonalize handles the size appropriately for now.
            # If latent_final is [B, C, L], diagonalize creates [B, 2C, L, L]. Pass this to decoder.
            pred_hic = self.decoder_2d(diag_input).squeeze(1) # Remove channel dim
            # Shape: [batch, hic_map_size, hic_map_size]
            outputs['hic'] = pred_hic

        if self.predict_1d:
            # Pass final latent sequence to 1D decoder
            pred_1d = self.decoder_1d(latent_final)
            # Shape: [batch, num_target_tracks, target_1d_length]
            outputs['1d'] = pred_1d

        if self.record_attn:
            outputs['attn_weights'] = attn_weights # Add weights to output dict if recorded

        # Return dictionary of predictions
        return outputs


    def move_feature_forward(self, x):
        '''
        Transpose between [batch, length, features] and [batch, features, length]
        '''
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        """
        Creates a 2D representation from 1D features for Hi-C prediction.
        Input shape: [batch, channels, length]
        Output shape: [batch, channels * 2, length, length] (approx)
        """
        # Check if length matches expected Hi-C dimension (e.g., 256)
        L = x.shape[-1]
        # If L is not the target size (e.g., 256), interpolation/resizing might be needed here or in the encoder/decoder design.
        # Assuming L is the intended dimension (e.g., 256)
        x_i = x.unsqueeze(3).repeat(1, 1, 1, L) # [B, C, L, L] - Copy C features along last dim
        x_j = x.unsqueeze(2).repeat(1, 1, L, 1) # [B, C, L, L] - Copy C features along third dim
        input_map = torch.cat([x_i, x_j], dim = 1) # [B, 2*C, L, L]
        return input_map

class ConvModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden = 256):
        super(ConvModel, self).__init__()
        #print('Initializing ConvModel')
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        self.decoder = blocks.Decoder2D(mid_hidden * 2)

    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

    def move_feature_forward(self, x):
        '''
        input dim:
        bs, img_len, feat
        to: 
        bs, feat, img_len
        '''
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map

class ConvTransModel(ConvModel):
    
    def __init__(self, num_genomic_features, mid_hidden = 256, record_attn = False, use_cross_attn = False):
        super(ConvTransModel, self).__init__(num_genomic_features)
        #print('Initializing ConvTransModel')
        if use_cross_attn:
            self.encoder = blocks.EncoderCrossAttn(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        else:
            self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn)
        self.decoder = blocks.Decoder2D(mid_hidden * 2)
        self.record_attn = record_attn
        self.use_cross_attn = use_cross_attn
    
    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        if self.use_cross_attn:
            x, cross_attn = self.encoder(x)
        else:
            x = self.encoder(x)
            cross_attn = None
        x = self.move_feature_forward(x)
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x = self.attn(x)
        x = self.move_feature_forward(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        if self.record_attn:
            return x, attn_weights, cross_attn
        else:
            return x
        
class ConvCrossAttnModel(nn.Module):
    # a model that uses cross attention across each of the genomic features instead of fusing them with convolutions
    def __init__(self, num_genomic_features, mid_hidden = 256, record_attn = False):
        super(ConvCrossAttnModel, self).__init__()
        


if __name__ == '__main__':
    main()
