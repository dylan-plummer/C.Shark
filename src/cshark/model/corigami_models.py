import math
import torch
import torch.nn as nn

import cshark.model.blocks as blocks


class CSharkUniversalModel(nn.Module):
    """
    A flexible, general-purpose model for genomic track prediction, aligned
    with the original C.Shark architectural patterns.

    This model accepts a variable set of input tracks (as a dictionary) and can
    be asked to predict any set of output tracks, enabling true de novo prediction
    for in-silico experiments.

    Architecture:
    1.  **Modal-Embedders**: Each input track (sequence, CTCF, etc.) is passed
        through its own dedicated CNN embedder (`blocks.Encoder`). This creates
        a latent representation for each modality.
    2.  **Modality-Tagging**: A unique, learnable "modality embedding" is added
        to each track's latent representation to identify its origin.
    3.  **Universal Transformer Encoder**: All tagged latent sequences are
        concatenated along the length dimension and processed by a single,
        powerful Transformer encoder. This creates a deeply contextualized
        "universal embedding" that captures all intra- and inter-track relationships.
    4.  **Aggregation & Decoding**:
        - The universal embedding is aggregated (by averaging) across all input
          modalities to create a single, fused latent representation.
        - This fused embedding is used by all decoders.
        - The Hi-C head uses the existing `diagonalize` and `Decoder2D` blocks.
        - The 1D track heads use the existing `Decoder1D` block, which performs
          the necessary upsampling to match `target_1d_length`.
    """
    def __init__(self, 
                 input_track_names: list,
                 all_track_names: list,
                 transformer_hidden_dim=64,
                 dim_feedforward=64,
                 num_transformer_layers=4,
                 target_mat_size=256,
                 target_1d_length=8192,
                 diploid=False,
                 predict_hic=True,
                 record_attn=False,
                 activation_1d=None,
                 seq_filter_size=15,
                 epi_filter_size=7,
                 **kwargs):
        super().__init__()
        self.input_track_names = input_track_names
        self.all_track_names = all_track_names
        self.transformer_hidden_dim = transformer_hidden_dim
        self.predict_hic = predict_hic
        self.record_attn = record_attn

        # --- 1. Modal Embedders ---
        embedder_blocks = 11 if target_mat_size == 512 else 12

        # Sequence Embedder (handles diploid case)
        self.seq_embedder = blocks.Encoder(
            in_channel=10 if diploid else 5,
            start_filter_size=seq_filter_size,
            output_size=transformer_hidden_dim,
            num_blocks=embedder_blocks
        )

        # A dictionary of embedders for each possible 1D track
        self.track_embedders = nn.ModuleDict({
            name: blocks.Encoder(in_channel=1, 
                                 start_filter_size=epi_filter_size,
                                 output_size=transformer_hidden_dim, 
                                 num_blocks=embedder_blocks)
            for name in input_track_names
        })

        # --- 2. Modality-Tagging and Universal Encoder ---
        self.modality_embeddings = nn.Parameter(torch.randn(1 + len(all_track_names), transformer_hidden_dim))
        self.pos_encoder = blocks.PositionalEncoding(transformer_hidden_dim, max_len=2048)
        encoder_layers = blocks.TransformerLayer(transformer_hidden_dim, nhead=4, dropout=0.1,
                                                 dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = blocks.TransformerEncoder(encoder_layers,
                                                             num_layers=num_transformer_layers,
                                                             record_attn=record_attn)

        # --- 3. Decoders (Prediction Heads) ---
        if self.predict_hic:
            self.decoder_2d = blocks.Decoder2D(transformer_hidden_dim * 2,
                                               hidden=transformer_hidden_dim * 2)

        # 1D decoders using your upsampling `Decoder1D` block
        num_upsample_blocks = int(math.log2(target_1d_length // target_mat_size))
        self.decoder_1d_heads = nn.ModuleDict({
            name: blocks.Decoder1D(
                num_target_tracks=1,
                latent_dim=transformer_hidden_dim,
                target_length=target_1d_length,
                num_upsample_blocks=num_upsample_blocks
            ) for name in all_track_names
        })
        self.activation_1d = activation_1d
        self.final_activation = None
        if activation_1d is not None:
            if activation_1d == 'relu':
                self.final_activation = nn.ReLU()
            elif activation_1d == 'sigmoid':
                self.final_activation = nn.Sigmoid()
            elif activation_1d == 'tanh':
                self.final_activation = nn.Tanh()
            elif activation_1d == 'softplus':
                self.final_activation = nn.Softplus()
            else:
                raise ValueError(f"Unsupported activation_1d: {activation_1d}")

    def forward(self, input_dict: dict, predict_tracks: list):
        """
        Args:
            input_dict (dict): Keys are track names (e.g., 'seq', 'ctcf'),
                               values are tensors. 'seq' is [B, C_seq, L],
                               other tracks are [B, 1, L].
            predict_tracks (list): List of track names to predict, e.g., ['hic', 'atac'].
        """
        all_embeddings = []
        embedding_segments = {}
        track_name_map = {name: i + 1 for i, name in enumerate(self.all_track_names)}

        # Process sequence
        if 'seq' in input_dict:
            seq_in = self.move_feature_forward(input_dict['seq']).float()  # Ensure input is float
            seq_latent = self.seq_embedder(seq_in).transpose(1, 2)
            seq_embed = seq_latent + self.modality_embeddings[0]
            all_embeddings.append(seq_embed)
            embedding_segments['seq'] = seq_embed

        # Process other 1D tracks
        for name, tensor in input_dict.items():
            if name == 'seq':
                continue
            track_latent = self.track_embedders[name](self.move_feature_forward(tensor).float()).transpose(1, 2)
            track_embed = track_latent + self.modality_embeddings[track_name_map[name]]
            all_embeddings.append(track_embed)
            embedding_segments[name] = track_embed

        # Create single sequence for the Universal Transformer
        full_sequence = torch.cat(all_embeddings, dim=1)
        full_sequence = self.pos_encoder(full_sequence)

        transformer_output = self.transformer_encoder(full_sequence)
        universal_embedding = transformer_output[0] if self.record_attn else transformer_output
        attn_weights = transformer_output[1] if self.record_attn else None

        # --- Aggregation and Decoding ---
        outputs = {'1d': {}, 'hic': None, 'attn_weights': attn_weights}

        # Reconstruct the different parts of the universal embedding to aggregate them
        current_pos = 0
        reconstructed_segments = []
        for name in embedding_segments.keys():
            segment_len = embedding_segments[name].shape[1]
            reconstructed_segments.append(universal_embedding[:, current_pos:current_pos + segment_len, :])
            current_pos += segment_len
        
        # Aggregate (average) across modalities to get a single fused latent representation
        stacked_segments = torch.stack(reconstructed_segments, dim=1)
        aggregated_latent = torch.mean(stacked_segments, dim=1) # Shape: [B, L', D]

        # Use the single aggregated embedding for all decoders
        latent_for_decoding = aggregated_latent.transpose(1, 2) # Shape: [B, D, L']

        if self.predict_hic and 'hic' in predict_tracks:
            diag_input = self.diagonalize(latent_for_decoding)
            outputs['hic'] = self.decoder_2d(diag_input).squeeze(1)

        # Predict all requested 1D tracks
        for track_name in predict_tracks:
            if track_name in self.decoder_1d_heads:
                # The Decoder1D block handles upsampling from L' to target_1d_length
                pred_1d = self.decoder_1d_heads[track_name](latent_for_decoding) # Output: [B, target_L, 1]
                if self.final_activation is not None:
                    pred_1d = self.final_activation(pred_1d)
                outputs['1d'][track_name] = pred_1d.squeeze(-1) # Output: [B, target_L]

        return outputs
    
    def move_feature_forward(self, x):
        '''
        Transpose between [batch, length, features] and [batch, features, length]
        '''
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        L = x.shape[-1]
        x_i = x.unsqueeze(3).repeat(1, 1, 1, L)
        x_j = x.unsqueeze(2).repeat(1, 1, L, 1)
        input_map = torch.cat([x_i, x_j], dim=1)
        return input_map


class MultiTaskConvTransModel(nn.Module): # Renamed for clarity
    """
    Predicts both 2D Hi-C maps and 1D tracks.
    """
    def __init__(self, num_genomic_features,
                 num_target_tracks=0, # Number of 1D tracks to predict
                 conditioning_vec_size=0, # Size of conditioning vector to concatenate
                 mid_hidden = 256,    # Latent dimension size
                 predict_hic = True,  # Whether to include the Hi-C prediction head
                 predict_1d = False,  # Whether to include the 1D prediction head
                 recon_1d = True, # Whether to reconstruct the 1D input features
                 diploid=False,
                 seq_filter_size = 3, # Filter size for sequence features
                 epi_filter_size = 5,
                 use_seq_attn = True, # Whether to use attention on sequence features before concatenation
                 target_mat_size = 256, # Expected size of the Hi-C map (e.g., 256x256)
                 target_1d_length = 8192, # Expected output length for 1D tracks
                 encoder_downsample_factor = 2**7, # Total downsampling from encoder (e.g., 13 blocks * stride 2)
                 activation_1d=None,
                 record_attn = False):
        super(MultiTaskConvTransModel, self).__init__()

        #print(f'Initializing MultiTaskConvTransModel with mid_hidden={mid_hidden}, predict_hic={predict_hic}, predict_1d={predict_1d}, recon_1d={recon_1d}, num_target_tracks={num_target_tracks}')

        if not predict_hic and not predict_1d:
            raise ValueError("Model must be configured to predict at least Hi-C or 1D tracks.")
        if predict_1d and num_target_tracks <= 0:
            raise ValueError("If predict_1d is True, num_target_tracks must be > 0.")

        # print(f'Initializing MultiTaskConvTransModel:')
        # print(f'  Predicting Hi-C: {predict_hic}')
        # print(f'  Predicting 1D Tracks: {predict_1d}')
        # if predict_1d:
        #     print(f'  Number of target 1D tracks: {num_target_tracks}')

        self.predict_hic = predict_hic
        self.predict_1d = predict_1d
        self.recon_1d = recon_1d
        self.num_target_tracks = num_target_tracks
        self.conditioning_vec_size = conditioning_vec_size
        self.record_attn = record_attn
        self.encoder_downsample_factor = encoder_downsample_factor
        self.use_seq_attn = use_seq_attn
        self.target_mat_size = target_mat_size
        self.target_1d_length = target_1d_length
        # 512 -> 11 256 -> 12 128 -> 13
        self.num_blocks = 11 if target_mat_size == 512 else (12 if target_mat_size == 256 else 12)
        print(f"Using {self.num_blocks} encoder blocks based on target_mat_size={target_mat_size}.")

        # --- Encoder ---
        # Takes sequence (5) + genomic features
        self.encoder = blocks.EncoderSplit(num_genomic_features, hidden = mid_hidden, output_size = mid_hidden, 
                                           num_blocks = self.num_blocks, num_bases=10 if diploid else 5,
                                           epi_filter_size=epi_filter_size,
                                           seq_filter_size=seq_filter_size,)
        # Output: [batch, mid_hidden, reduced_length]

        if conditioning_vec_size > 0:
            print(f"Using conditioning vector of size {conditioning_vec_size}.")
            self.condition_mlp = nn.Sequential(
                nn.Linear(conditioning_vec_size, 2048),
                nn.GELU(),
                nn.Linear(2048, mid_hidden)
            )

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
            #print(f"1D Decoder using latent dim={mid_hidden}{target_1d_length}")
            # number of upsamples to go from 256 to target_1d_length
            num_upsample_blocks = int(math.log2(target_1d_length // target_mat_size)) # Number of upsample blocks based on target length
            if self.recon_1d:
                self.decoder_1d = blocks.Decoder1D(num_target_tracks = self.num_target_tracks,
                                                num_upsample_blocks=num_upsample_blocks,
                                                latent_dim=mid_hidden,
                                                target_length=self.target_1d_length)
                self.decoder_1d_seq = None 
            else:
                self.decoder_1d = None
                if self.num_target_tracks - num_genomic_features > 0:
                    self.decoder_1d = blocks.Decoder1D(num_target_tracks = self.num_target_tracks - num_genomic_features,
                                                    num_upsample_blocks=num_upsample_blocks,
                                                    latent_dim=mid_hidden,
                                                    target_length=self.target_1d_length)
                # If not reconstructing 1D features, we need to predict them from sequence features
                if use_seq_attn:
                    self.seq_attn = blocks.AttnModule(hidden = mid_hidden // 2, record_attn = False, layers=4)
                self.decoder_1d_seq = blocks.Decoder1D(num_target_tracks = num_genomic_features,
                                                       num_upsample_blocks=num_upsample_blocks,
                                                       latent_dim=mid_hidden // 2,
                                                       target_length=self.target_1d_length)
            # Output: [batch, num_target_tracks, target_1d_length]
    
        self.activation_1d = activation_1d
        self.final_activation = None
        if activation_1d is not None:
            if activation_1d == 'relu':
                self.final_activation = nn.ReLU()
            elif activation_1d == 'sigmoid':
                self.final_activation = nn.Sigmoid()
            elif activation_1d == 'tanh':
                self.final_activation = nn.Tanh()
            elif activation_1d == 'softplus':
                self.final_activation = nn.Softplus()
            else:
                raise ValueError(f"Unsupported activation_1d: {activation_1d}")

    def forward(self, x, conditioning_vec=None):
        '''
        Input feature x: [batch_size, length, feature_dim (5 + num_genomic_features)]
        '''
        # 1. Permute to [batch, features, length] for Conv1d
        x = self.move_feature_forward(x).float()
        # Shape: [batch, 5 + num_genomic_features, length]

        # 2. Encode
        if self.recon_1d:  # use full features for 1d prediction (input tracks are reconstructed)
            latent_seq = self.encoder(x)
        else: # only use full features for Hi-C and other 1d predictions (input tracks are predicted only from sequence features)
            latent_seq, seq_feats = self.encoder(x, return_seq_feats=True)

        # Shape: [batch, mid_hidden, reduced_length]

        # If conditioning vector is provided, integrate it
        if conditioning_vec is not None:
            conditioning_vec = self.condition_mlp(conditioning_vec)
            latent_seq = latent_seq + conditioning_vec.unsqueeze(2)

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
        #print(f"Final latent shape: {latent_final.shape}")
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
            if self.recon_1d:
                # Pass final latent sequence to 1D decoder
                pred_1d = self.decoder_1d(latent_final)
            else:
                if self.use_seq_attn:
                    seq_feats_permuted = self.move_feature_forward(seq_feats) # Use sequence features for 1D prediction
                    seq_feats_transformed = self.seq_attn(seq_feats_permuted) # Apply attention to sequence features
                    seq_feats_final = self.move_feature_forward(seq_feats_transformed) # Permute back
                pred_1d_inputs = self.decoder_1d_seq(seq_feats_final) # Use sequence features for 1D prediction
                if self.decoder_1d is not None:
                    pred_1d = self.decoder_1d(latent_final)
                    #print(f"Predicted 1D tracks shape: {pred_1d.shape}")
                    pred_1d = torch.cat([pred_1d_inputs, pred_1d], dim=2) # Concatenate sequence features with predicted 1D tracks
                else:
                    pred_1d = pred_1d_inputs
            if self.final_activation is not None:
                pred_1d = self.final_activation(pred_1d)
            # Shape: [batch, num_target_tracks, target_1d_length]
            outputs['1d'] = pred_1d
        else:
            outputs['1d'] = None # No 1D prediction if not configured

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


class MultiTaskConvTransModelOld(nn.Module): # Renamed for clarity
    """
    Predicts both 2D Hi-C maps and 1D tracks.
    """
    def __init__(self, num_genomic_features,
                 num_target_tracks=0, # Number of 1D tracks to predict
                 mid_hidden = 256,    # Latent dimension size
                 predict_hic = True,  # Whether to include the Hi-C prediction head
                 predict_1d = False,  # Whether to include the 1D prediction head
                 recon_1d = True, # Whether to reconstruct the 1D input features
                 diploid=False,
                 seq_filter_size = 3, # Filter size for sequence features
                 epi_filter_size = 5,
                 target_mat_size = 256, # Expected size of the Hi-C map (e.g., 256x256)
                 target_1d_length = 2048, # Expected output length for 1D tracks
                 encoder_downsample_factor = 2**7, # Total downsampling from encoder (e.g., 13 blocks * stride 2)
                 record_attn = False):
        super(MultiTaskConvTransModelOld, self).__init__()

        if not predict_hic and not predict_1d:
            raise ValueError("Model must be configured to predict at least Hi-C or 1D tracks.")
        if predict_1d and num_target_tracks <= 0:
            raise ValueError("If predict_1d is True, num_target_tracks must be > 0.")

        # print(f'Initializing MultiTaskConvTransModel:')
        # print(f'  Predicting Hi-C: {predict_hic}')
        # print(f'  Predicting 1D Tracks: {predict_1d}')
        # if predict_1d:
        #     print(f'  Number of target 1D tracks: {num_target_tracks}')

        self.predict_hic = predict_hic
        self.predict_1d = predict_1d
        self.recon_1d = recon_1d
        self.num_target_tracks = num_target_tracks
        self.record_attn = record_attn
        self.encoder_downsample_factor = encoder_downsample_factor
        self.target_mat_size = target_mat_size
        self.target_1d_length = target_1d_length
        self.num_blocks = 11 if target_mat_size == 512 else 12 # Number of blocks in the encoder

        # --- Encoder ---
        # Takes sequence (5) + genomic features
        self.encoder = blocks.EncoderSplit(num_genomic_features, hidden = mid_hidden, output_size = mid_hidden, 
                                           num_blocks = self.num_blocks, num_bases=10 if diploid else 5,
                                           seq_filter_size=seq_filter_size, epi_filter_size=epi_filter_size)
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
            #print(f"1D Decoder using latent dim={mid_hidden}{target_1d_length}")
            if self.recon_1d:
                self.decoder_1d = blocks.Decoder1D(num_target_tracks = self.num_target_tracks,
                                                num_upsample_blocks=2 if target_mat_size == 512 else 3,
                                                latent_dim=mid_hidden,
                                                target_length=self.target_1d_length)
                self.decoder_1d_seq = None 
            else:
                self.decoder_1d = blocks.Decoder1D(num_target_tracks = self.num_target_tracks - num_genomic_features,
                                                num_upsample_blocks=2 if target_mat_size == 512 else 3,
                                                latent_dim=mid_hidden,
                                                target_length=self.target_1d_length)
                # If not reconstructing 1D features, we need to predict them from sequence features
                self.decoder_1d_seq = blocks.Decoder1D(num_target_tracks = num_genomic_features,
                                                       num_upsample_blocks=2 if target_mat_size == 512 else 3,
                                                       latent_dim=128,
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
        if self.recon_1d:  # use full features for 1d prediction (input tracks are reconstructed)
            latent_seq = self.encoder(x)
        else: # only use full features for Hi-C and other 1d predictions (input tracks are predicted only from sequence features)
            latent_seq, seq_feats = self.encoder(x, return_seq_feats=True)

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
        #print(f"Final latent shape: {latent_final.shape}")
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
            if self.recon_1d:
                # Pass final latent sequence to 1D decoder
                pred_1d = self.decoder_1d(latent_final)
            else:
                pred_1d_inputs = self.decoder_1d_seq(seq_feats) # Use sequence features for 1D prediction
                pred_1d = self.decoder_1d(latent_final)
                pred_1d = torch.cat([pred_1d_inputs, pred_1d], dim=2) # Concatenate sequence features with predicted 1D tracks
            # Shape: [batch, num_target_tracks, target_1d_length]
            outputs['1d'] = pred_1d
        else:
            outputs['1d'] = None # No 1D prediction if not configured

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
            self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, 
                                               epi_filter_size=3,
                                               num_blocks = 12)
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