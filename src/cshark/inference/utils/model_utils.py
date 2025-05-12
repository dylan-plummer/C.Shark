import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cshark.model.corigami_models as corigami_models


def get_1d_track_names(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    try:
        target_tracks = checkpoint['hyper_parameters']['output_features']
        if isinstance(target_tracks, list):
            target_tracks = [track.replace('_norm', '') for track in target_tracks]
        else:
            target_tracks = [target_tracks.replace('_norm', '')]
        return target_tracks
    except KeyError:
        return []

def load_default(model_path, record_attn=False, 
                 num_genomic_features=2, mat_size=256,
                 mid_hidden=256, 
                 diploid=False,
                 model_name='ConvTransModel'):
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
                              predict_1d=False)
            load_checkpoint(model, model_path)
        except Exception as e:  # new C.Shark checkpoint (with 1D tracks)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            num_target_tracks = len(checkpoint['hyper_parameters']['output_features'])
            model = get_model('MultiTaskConvTransModel', mid_hidden, 
                              num_genomic_features=num_genomic_features, 
                              mat_size=mat_size,
                              record_attn=record_attn, 
                              diploid=diploid,
                              num_target_tracks=num_target_tracks, 
                              predict_1d=True)
            load_checkpoint(model, model_path)
    return model

def get_model(model_name, mid_hidden, num_genomic_features=2, mat_size=256, 
              diploid=False,
              num_target_tracks=0, predict_1d=False,
              record_attn=False):
    ModelClass = getattr(corigami_models, model_name)
    if model_name == 'MultiTaskConvTransModel':
        model = ModelClass(num_genomic_features, 
                           num_target_tracks=num_target_tracks, 
                           mid_hidden=mid_hidden, 
                           predict_1d=predict_1d,
                           target_mat_size=mat_size, 
                           diploid=diploid,
                           record_attn=record_attn)
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
