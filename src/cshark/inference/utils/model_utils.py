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
        target_tracks = checkpoint['hyper_parameters']['output_features'].copy()
        all_tracks = checkpoint['hyper_parameters']['input_features'].copy()
        input_tracks = checkpoint['hyper_parameters']['input_features'].copy()
        for track in target_tracks:
            if track not in all_tracks:
                all_tracks.append(track)
        if target_tracks is not None:
            if isinstance(target_tracks, list):
                target_tracks = [track.replace('_norm', '') for track in target_tracks]
            else:
                target_tracks = [target_tracks.replace('_norm', '')]
        if all_tracks is not None:
            if isinstance(all_tracks, list):
                all_tracks = [track.replace('_norm', '') for track in all_tracks]
            else:
                all_tracks = [all_tracks.replace('_norm', '')]
        if input_tracks is not None:
            if isinstance(input_tracks, list):
                input_tracks = [track.replace('_norm', '') for track in input_tracks]
            else:
                input_tracks = [input_tracks.replace('_norm', '')]
        return all_tracks, target_tracks, input_tracks
    except KeyError:
        return []

def load_default(model_path, record_attn=False, 
                 num_genomic_features=2, mat_size=256,
                 seq_filter_size=3,
                 recon_1d=False,
                 mid_hidden=256, 
                 diploid=False,
                 model_name='ConvTransModel'):
    try:  # check if universal model
        all_track_names, target_tracks, input_tracks = get_all_track_names(model_path)
        # print(f'All track names: {all_track_names}')
        # print(f'Target tracks: {target_tracks}')
        # print(f'Input tracks: {input_tracks}')
        num_target_tracks = len(target_tracks)
        model = get_model('CSharkUniversalModel', mid_hidden, 
                        num_genomic_features=num_genomic_features, 
                        mat_size=mat_size,
                        record_attn=record_attn, 
                        diploid=diploid,
                        num_target_tracks=num_target_tracks, 
                        target_1d_length=4096,
                        seq_filter_size=seq_filter_size,
                        recon_1d=recon_1d,
                        predict_1d=True,
                        input_track_names=input_tracks,
                        all_track_names=all_track_names)
        load_checkpoint(model, model_path)
    except Exception as e:  # fallback to old models
        try:  # old C.Origami checkpoint
            model = get_model(model_name, mid_hidden, 
                            num_genomic_features=num_genomic_features, 
                            mat_size=mat_size, 
                            record_attn=record_attn)
            load_checkpoint(model, model_path)
        except Exception as e:
            try:  # new C.Shark checkpoint (no 1D tracks)
                model = get_model('MultiTaskConvTransModel', mid_hidden, 
                                num_genomic_features=num_genomic_features, 
                                mat_size=mat_size,
                                record_attn=record_attn, 
                                diploid=diploid,
                                num_target_tracks=0, 
                                seq_filter_size=seq_filter_size,
                                recon_1d=recon_1d,
                                predict_1d=False)
                load_checkpoint(model, model_path)
            except Exception as e:  # new C.Shark checkpoint (with 1D tracks)
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    num_target_tracks = len(checkpoint['hyper_parameters']['output_features'])
                    model = get_model('MultiTaskConvTransModel', mid_hidden, 
                                    num_genomic_features=num_genomic_features, 
                                    mat_size=mat_size,
                                    record_attn=record_attn, 
                                    diploid=diploid,
                                    num_target_tracks=num_target_tracks, 
                                    seq_filter_size=seq_filter_size,
                                    recon_1d=recon_1d,
                                    predict_1d=True)
                    load_checkpoint(model, model_path)
                except Exception as e:  # fallback to older 1D track model
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    num_target_tracks = len(checkpoint['hyper_parameters']['output_features'])
                    model = get_model('MultiTaskConvTransModelOld', mid_hidden, 
                                    num_genomic_features=num_genomic_features, 
                                    mat_size=mat_size,
                                    record_attn=record_attn, 
                                    diploid=diploid,
                                    num_target_tracks=num_target_tracks, 
                                    seq_filter_size=seq_filter_size,
                                    epi_filter_size=3,
                                    target_1d_length=2048,
                                    recon_1d=recon_1d,
                                    predict_1d=True)
                    #print(model)
                    load_checkpoint(model, model_path)
    return model

def get_model(model_name, mid_hidden, num_genomic_features=2, mat_size=256, 
              diploid=False,
              num_target_tracks=0, predict_1d=False,
              seq_filter_size=3,
              epi_filter_size=5,
              use_seq_attn=True,
              target_1d_length=8192,
              recon_1d=False,
              record_attn=False,
              input_track_names=None,
              all_track_names=None):
    ModelClass = getattr(corigami_models, model_name)
    if model_name == 'MultiTaskConvTransModel':
        model = ModelClass(num_genomic_features, 
                           num_target_tracks=num_target_tracks, 
                           mid_hidden=mid_hidden, 
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
                           mid_hidden=mid_hidden, 
                           predict_hic=True,
                           predict_1d=predict_1d,
                           target_mat_size=mat_size, 
                           diploid=diploid,
                           seq_filter_size=seq_filter_size,
                           epi_filter_size=epi_filter_size,
                           use_seq_attn=use_seq_attn,
                           target_1d_length=target_1d_length,
                           recon_1d=recon_1d,
                           record_attn=record_attn)
        #print(model.decoder_1d_heads)
        
    else:
        model = ModelClass(num_genomic_features, mid_hidden = mid_hidden, record_attn=record_attn)
    return model

def load_checkpoint(model, model_path):
    #print('Loading weights')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    #print(checkpoint)
    model_weights = checkpoint['state_dict']

    # Edit keys
    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()
    return model

if __name__ == '__main__':
    main()
