import sys
from tabnanny import check
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cshark.model.corigami_models as corigami_models


def get_1d_track_names(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    try:
        target_tracks = checkpoint['hyper_parameters']['output_features']
        if target_tracks is not None:
            if isinstance(target_tracks, list):
                target_tracks = [track.replace('_norm', '') for track in target_tracks]
            else:
                target_tracks = [target_tracks.replace('_norm', '')]
        return target_tracks
    except KeyError:
        return []
    

def get_all_track_names(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    try:
        target_tracks = list(checkpoint['hyper_parameters']['output_features']) if checkpoint['hyper_parameters']['output_features'] else []
        all_tracks = list(checkpoint['hyper_parameters']['input_features']) if checkpoint['hyper_parameters']['input_features'] else []
        input_tracks = list(checkpoint['hyper_parameters']['input_features']) if checkpoint['hyper_parameters']['input_features'] else []
        
        if target_tracks is not None:
            for track in target_tracks:
                if track not in all_tracks:
                    all_tracks.append(track)
            if isinstance(target_tracks, list):
                target_tracks = [track.replace('_norm', '') for track in target_tracks]
            else:
                target_tracks = [target_tracks.replace('_norm', '')]
        else:
            target_tracks = []
        if all_tracks is not None:
            if isinstance(all_tracks, list):
                all_tracks = [track.replace('_norm', '') for track in all_tracks]
            else:
                all_tracks = [all_tracks.replace('_norm', '')]
        else:
            all_tracks = []
        if input_tracks is not None:
            if isinstance(input_tracks, list):
                input_tracks = [track.replace('_norm', '') for track in input_tracks]
            else:
                input_tracks = [input_tracks.replace('_norm', '')]
        else:
            input_tracks = []
        return all_tracks, target_tracks, input_tracks
    except KeyError:
        return []

def _extract_main_model_state_dict(state_dict):
    """Return the main Hi-C model's weights from a (possibly unified) checkpoint.

    A hierarchical end-to-end checkpoint stores three sub-modules:
    ``model.*`` (Hi-C), ``input_pred_model.*`` (RAD21) and ``enformer.*``.
    This loader builds only the Hi-C model, so we keep the ``model.*`` keys
    (stripping just the leading prefix) and drop the rest.  Using
    ``str.replace('model.', '')`` would be wrong: it also rewrites
    ``input_pred_model.`` -> ``input_pred_`` and pulls in foreign weights.

    Checkpoints whose keys are not nested under ``model.`` (older standalone
    exports) are returned unchanged.
    """
    prefix = 'model.'
    main_keys = [k for k in state_dict if k.startswith(prefix)]
    if not main_keys:
        return dict(state_dict)
    drop_prefixes = ('input_pred_model.', 'enformer.')
    return {k[len(prefix):]: v for k, v in state_dict.items()
            if k.startswith(prefix) and not k.startswith(drop_prefixes)}


def load_default(model_path, record_attn=False,
                 num_genomic_features=2, mat_size=256,
                 seq_filter_size=3,
                 recon_1d=False,
                 mid_hidden=256, 
                 target_1d_length=8192,
                 diploid=False,
                 no_hic=False,
                 conditioning_vec_size=0,
                 model_name='ConvTransModel'):
    # Pre-load checkpoint once to detect model type and infer dimensions
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ckpt = torch.load(model_path, map_location=_device, weights_only=False)
    _sd = _ckpt['state_dict']
    _hp = _ckpt.get('hyper_parameters', {})
    # Isolate the main Hi-C model's weights.  In a unified hierarchical
    # checkpoint the state dict also holds `input_pred_model.*` and `enformer.*`;
    # those must be dropped here (this loader only builds the main model).
    _sd_clean = _extract_main_model_state_dict(_sd)
    # Check if this is truly a CSharkUniversalModel by looking at state_dict keys
    _is_universal = any('modality_embeddings' in k or 'track_embedders' in k for k in _sd_clean)
    # Infer the latent width (mid_hidden) straight from the weights so callers
    # don't have to pass a matching --latent_size.  EncoderSplit.conv_end is
    # nn.Conv1d(256, mid_hidden, 1) -> shape[0] == mid_hidden.
    _cend_key = next((k for k in _sd_clean if k.endswith('encoder.conv_end.weight')), None)
    if _cend_key is not None:
        mid_hidden = _sd_clean[_cend_key].shape[0]
    # Match training: softplus only when bigwig_log_transform is False
    _bigwig_log = _hp.get('bigwig_log_transform', True)
    _activation_1d = 'softplus' if not _bigwig_log else None
    # Infer architecture params from checkpoint weights
    _epi_key = next((k for k in _sd_clean if 'conv_start_epi.0.weight' in k), None)
    if _epi_key is not None:
        _num_gf = _sd_clean[_epi_key].shape[1]  # num_genomic_features
        _epi_fs = _sd_clean[_epi_key].shape[2]   # epi_filter_size
    else:
        _num_gf = num_genomic_features
        _epi_fs = 3
    _seq_key = next((k for k in _sd_clean if 'conv_start_seq.0.weight' in k), None)
    if _seq_key is not None:
        _seq_fs = _sd_clean[_seq_key].shape[2]   # seq_filter_size
    else:
        _seq_fs = seq_filter_size
    # Infer mat_size from number of encoder res_blocks
    _max_rb = max((int(k.split('.')[2]) for k in _sd_clean if k.startswith('encoder.res_blocks_seq.')), default=11)
    _mat = 512 if _max_rb < 11 else _hp.get('mat_size', mat_size)
    # Infer target_1d_length from decoder_1d upsample blocks
    _max_us = max((int(k.split('.')[2]) for k in _sd_clean if k.startswith('decoder_1d.upsample_blocks.')), default=-1)
    _t1d = int(_mat * (2 ** (_max_us + 1))) if _max_us >= 0 else target_1d_length
    del _ckpt

    if _is_universal:
        try:
            all_track_names, target_tracks, input_tracks = get_all_track_names(model_path)
            num_target_tracks = len(target_tracks)
            # Infer architecture dimensions from checkpoint weights
            _mod_key = next((k for k in _sd_clean if 'modality_embeddings' in k), None)
            _thd = _sd_clean[_mod_key].shape[-1] if _mod_key else mid_hidden
            _ff_key = next((k for k in _sd_clean if 'linear1.weight' in k), None)
            _dff = _sd_clean[_ff_key].shape[0] if _ff_key else 64
            _mat = _hp.get('mat_size', mat_size)
            _predict_hic = _hp.get('predict_hic', True)
            try:
                model = get_model('CSharkUniversalModel', _thd, 
                                num_genomic_features=num_genomic_features, 
                                mat_size=_mat,
                                record_attn=record_attn, 
                                diploid=diploid,
                                num_target_tracks=num_target_tracks, 
                                target_1d_length=target_1d_length,
                                seq_filter_size=15,
                                epi_filter_size=7,
                                dim_feedforward=_dff,
                                recon_1d=recon_1d,
                                predict_hic=_predict_hic,
                                predict_1d=True,
                                input_track_names=input_tracks,
                                all_track_names=all_track_names,
                                activation_1d=_activation_1d)
                load_checkpoint(model, model_path)
            except Exception as e:  # fallback to old universal model
                model = get_model('CSharkUniversalModel', _thd, 
                                num_genomic_features=num_genomic_features, 
                                mat_size=_mat,
                                record_attn=record_attn, 
                                diploid=diploid,
                                num_target_tracks=num_target_tracks, 
                                target_1d_length=8192,
                                seq_filter_size=3,
                                epi_filter_size=3,
                                dim_feedforward=32,
                                recon_1d=recon_1d,
                                predict_hic=_predict_hic,
                                predict_1d=True,
                                input_track_names=input_tracks,
                                all_track_names=all_track_names,
                                activation_1d=_activation_1d)
                load_checkpoint(model, model_path)
            return model
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint '{model_path}' detected as CSharkUniversalModel "
                f"(has modality_embeddings/track_embedders keys) but failed to load: {e}"
            ) from e

    _load_errors = []  # collect concise error from each attempt

    try:  # old C.Origami checkpoint
        model = get_model(model_name, mid_hidden, 
                        num_genomic_features=_num_gf, 
                        mat_size=_mat, 
                        epi_filter_size=_epi_fs,
                        record_attn=record_attn)
        load_checkpoint(model, model_path)
        return model
    except Exception as e:
        _load_errors.append(f"{model_name}: {e}")

    try:  # new C.Shark checkpoint (no 1D tracks)
        model = get_model('MultiTaskConvTransModel', mid_hidden, 
                        num_genomic_features=_num_gf, 
                        target_1d_length=_t1d,
                        conditioning_vec_size=conditioning_vec_size,
                        mat_size=_mat,
                        record_attn=record_attn, 
                        diploid=diploid,
                        num_target_tracks=0, 
                        seq_filter_size=_seq_fs,
                        epi_filter_size=_epi_fs,
                        recon_1d=recon_1d,
                        predict_hic=not no_hic,
                        predict_1d=False)
        load_checkpoint(model, model_path)
        return model
    except Exception as e:
        _load_errors.append(f"MultiTaskConvTransModel(no 1D): {e}")

    try:  # new C.Shark checkpoint (with 1D tracks)
        num_target_tracks = len(_hp.get('output_features', []))
        model = get_model('MultiTaskConvTransModel', mid_hidden, 
                        num_genomic_features=_num_gf, 
                        target_1d_length=_t1d,
                        conditioning_vec_size=conditioning_vec_size,
                        mat_size=_mat,
                        record_attn=record_attn, 
                        diploid=diploid,
                        num_target_tracks=num_target_tracks, 
                        seq_filter_size=_seq_fs,
                        epi_filter_size=_epi_fs,
                        recon_1d=recon_1d,
                        predict_hic=not no_hic,
                        predict_1d=True)
        load_checkpoint(model, model_path)
        return model
    except Exception as e:
        _load_errors.append(f"MultiTaskConvTransModel(1D): {e}")

    try:  # fallback to older 1D track model
        _out_feats = _hp.get('output_features', None)
        if _out_feats is None:
            num_target_tracks = 0
        else:
            num_target_tracks = len(_out_feats)
        model = get_model('MultiTaskConvTransModelOld', mid_hidden, 
                        num_genomic_features=_num_gf, 
                        mat_size=_mat,
                        record_attn=record_attn, 
                        diploid=diploid,
                        num_target_tracks=num_target_tracks, 
                        seq_filter_size=_seq_fs,
                        epi_filter_size=_epi_fs,
                        target_1d_length=2048,
                        recon_1d=recon_1d,
                        predict_1d=num_target_tracks>0)
        load_checkpoint(model, model_path)
        return model
    except Exception as e:
        _load_errors.append(f"MultiTaskConvTransModelOld: {e}")

    error_summary = "\n  ".join(_load_errors)
    raise RuntimeError(
        f"Failed to load checkpoint '{model_path}' with any known model architecture.\n"
        f"  Inferred: num_genomic_features={_num_gf}, mat_size={_mat}, "
        f"epi_filter={_epi_fs}, seq_filter={_seq_fs}, universal={_is_universal}\n"
        f"  Tried:\n  {error_summary}"
    )

def get_model(model_name, mid_hidden, num_genomic_features=2, mat_size=256, 
              diploid=False,
              num_target_tracks=0, 
              conditioning_vec_size=0,
              predict_hic=True,
              predict_1d=False,
              seq_filter_size=3,
              epi_filter_size=5,
              dim_feedforward=64,
              use_seq_attn=True,
              target_1d_length=8192,
              recon_1d=False,
              record_attn=False,
              input_track_names=None,
              all_track_names=None,
              activation_1d=None):
    ModelClass = getattr(corigami_models, model_name)
    if model_name == 'MultiTaskConvTransModel':
        model = ModelClass(num_genomic_features, 
                           num_target_tracks=num_target_tracks, 
                           conditioning_vec_size=conditioning_vec_size,
                           mid_hidden=mid_hidden, 
                           predict_hic=predict_hic,
                           predict_1d=predict_1d,
                           target_mat_size=mat_size, 
                           diploid=diploid,
                           seq_filter_size=seq_filter_size,
                           epi_filter_size=epi_filter_size,
                           use_seq_attn=use_seq_attn,
                           target_1d_length=target_1d_length,
                           recon_1d=recon_1d,
                           record_attn=record_attn)
    elif model_name == 'MultiTaskConvTransModelOld':
        model = ModelClass(num_genomic_features, 
                           num_target_tracks=num_target_tracks, 
                           mid_hidden=mid_hidden, 
                           predict_1d=predict_1d,
                           target_mat_size=mat_size, 
                           diploid=diploid,
                           seq_filter_size=seq_filter_size,
                           epi_filter_size=epi_filter_size,
                           target_1d_length=target_1d_length,
                           recon_1d=recon_1d,
                           record_attn=record_attn)
    elif model_name == 'CSharkUniversalModel':

        model = ModelClass(input_track_names=input_track_names,     
                           all_track_names=all_track_names,
                           transformer_hidden_dim=mid_hidden, 
                           predict_hic=predict_hic,
                           predict_1d=predict_1d,
                           target_mat_size=mat_size, 
                           diploid=diploid,
                           seq_filter_size=seq_filter_size,
                           epi_filter_size=epi_filter_size,
                           dim_feedforward=dim_feedforward,
                           use_seq_attn=use_seq_attn,
                           target_1d_length=target_1d_length,
                           recon_1d=recon_1d,
                           record_attn=record_attn,
                           activation_1d=activation_1d)
        #print(model.decoder_1d_heads)
        
    else:
        model = ModelClass(num_genomic_features, mid_hidden = mid_hidden, record_attn=record_attn)
    return model

def load_checkpoint(model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Keep only the main Hi-C model weights; drop input_pred_model.* / enformer.*
    # that a unified hierarchical checkpoint also carries.
    model_weights = _extract_main_model_state_dict(checkpoint['state_dict'])
    result = model.load_state_dict(model_weights, strict=False)
    if result.missing_keys or result.unexpected_keys:
        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)
        msg = (f"State dict mismatch for {model.__class__.__name__}: "
               f"{n_missing} missing key(s), {n_unexpected} unexpected key(s).")
        if n_missing > 0:
            msg += f"\n  First missing: {result.missing_keys[0]}"
        if n_unexpected > 0:
            msg += f"\n  First unexpected: {result.unexpected_keys[0]}"
        raise RuntimeError(msg)
    model.eval()
    return model

if __name__ == '__main__':
    main()
