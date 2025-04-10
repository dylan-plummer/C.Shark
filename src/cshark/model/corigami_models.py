import torch
import torch.nn as nn

import cshark.model.blocks as blocks

class ConvModel(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden = 256):
        super(ConvModel, self).__init__()
        print('Initializing ConvModel')
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        self.decoder = blocks.Decoder(mid_hidden * 2)

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
        print('Initializing ConvTransModel')
        if use_cross_attn:
            self.encoder = blocks.EncoderCrossAttn(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        else:
            self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)
        self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn)
        self.decoder = blocks.Decoder(mid_hidden * 2)
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
