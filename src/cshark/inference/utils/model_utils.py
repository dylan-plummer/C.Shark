import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cshark.model.corigami_models as corigami_models

def load_default(model_path, record_attn=False, use_cross_attn=False, num_genomic_features=2, mid_hidden=256):
    model_name = 'ConvTransModel'
    model = get_model(model_name, mid_hidden, num_genomic_features=num_genomic_features, record_attn=record_attn, use_cross_attn=use_cross_attn)
    load_checkpoint(model, model_path)
    return model

def get_model(model_name, mid_hidden, num_genomic_features=2, record_attn=False, use_cross_attn=False):
    ModelClass = getattr(corigami_models, model_name)
    model = ModelClass(num_genomic_features, mid_hidden = mid_hidden, record_attn=record_attn, use_cross_attn=use_cross_attn)
    return model

def load_checkpoint(model, model_path):
    print('Loading weights')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_weights = checkpoint['state_dict']

    # Edit keys
    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()
    return model

if __name__ == '__main__':
    main()
