import jittor
import jittor.nn as nn
# import jittor.nn as F

import math
import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

################################################################################################
################################################################################################
################################################################################################

def compute_ap_score(pred_pos, pred_neg, neg_samples):
        y_pred = jittor.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu().detach()
        y_true = jittor.cat([jittor.ones_like(pred_pos), jittor.zeros_like(pred_neg)], dim=0).cpu().detach()
        acc = average_precision_score(y_true, y_pred)
        if neg_samples > 1:
            auc = jittor.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0)
            auc = 1 / (auc+1)
        else:
            auc = roc_auc_score(y_true, y_pred)
        return acc, auc 
    
################################################################################################
################################################################################################
################################################################################################


def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def reset_linear_parameters(linear_layer):
    nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
    if linear_layer.bias is not None:
        fan_in, _ = calculate_fan_in_and_fan_out(linear_layer.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear_layer.bias, -bound, bound)


def reset_laynorm_parameters(laynorm_layer):
    if laynorm_layer.elementwise_affine:
        # nn.init.one(laynorm_layer.weight)
        nn.init.constant_(laynorm_layer.weight, 1)
        if laynorm_layer.bias is not None:
            # nn.init.zero(laynorm_layer.bias)
            nn.init.constant_(laynorm_layer.bias, 0)

"""
Module: Time-encoder
"""

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((jittor.array(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(jittor.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @jittor.no_grad()
    def execute(self, t):
        output = jittor.cos(self.w(t.reshape((-1, 1))))
        return output



################################################################################################
################################################################################################
################################################################################################
"""
Module: MLP-Mixer
"""

class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        # self.linear_0.reset_parameters()
        reset_linear_parameters(self.linear_0)
        if self.use_single_layer==False:
            # self.linear_1.reset_parameters()
            reset_linear_parameters(self.linear_1)

    def execute(self, x):
        x = self.linear_0(x)
        x = nn.gelu(x)
        x = nn.dropout(x, p=self.dropout, is_train=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = nn.dropout(x, p=self.dropout, is_train=
                           self.training)
        return x

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, per_graph_size, dims, 
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 dropout=0, 
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')

        if 'token' in self.module_spec:
            self.token_layernorm = nn.LayerNorm(dims)
            self.token_forward = FeedForward(per_graph_size, token_expansion_factor, dropout, use_single_layer)
            
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        

    def reset_parameters(self):
        if 'token' in self.module_spec:
            # self.token_layernorm.reset_parameters()
            reset_laynorm_parameters(self.token_layernorm)
            self.token_forward.reset_parameters()

        if 'channel' in self.module_spec:
            # self.channel_layernorm.reset_parameters()
            reset_laynorm_parameters(self.channel_layernorm)
            self.channel_forward.reset_parameters()
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def execute(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)
        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x
    
class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims)
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        reset_linear_parameters(self.feat_encoder)
        
    def execute(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = jittor.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)

class MLPMixer(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 module_spec=None, use_single_layer=False
                ):
        super().__init__()
        self.per_graph_size = per_graph_size

        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers
        self.mixer_blocks = jittor.nn.ModuleList()
        for ell in range(num_layers):
            if module_spec is None:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=None, 
                               use_single_layer=use_single_layer)
                )
            else:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=module_spec[ell], 
                               use_single_layer=use_single_layer)
                )



        # init
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        # self.layernorm.reset_parameters()
        reset_laynorm_parameters(self.layernorm)
        # self.mlp_head.reset_parameters()
        reset_linear_parameters(self.mlp_head)


    def execute(self, edge_feats, edge_ts, batch_size, inds):
        # x :     [ batch_size, graph_size, edge_dims+time_dims]
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
        x = jittor.zeros((batch_size * self.per_graph_size, 
                         edge_time_feats.size(1)))
        x[inds] = x[inds] + edge_time_feats     
        x = jittor.split(x, self.per_graph_size)
        x = jittor.stack(x)
        
        # apply to original feats
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x)
        x = self.layernorm(x)
        x = jittor.mean(x, dim=1)
        x = self.mlp_head(x)
        return x
    
################################################################################################
################################################################################################
################################################################################################

"""
Edge predictor
"""

class EdgePredictor_per_node(jittor.nn.Module):
    """
    out = linear(src_node_feats) + linear(dst_node_feats)
    out = ReLU(out)
    """
    def __init__(self, dim_in_time, dim_in_node):
        super().__init__()

        self.dim_in_time = dim_in_time
        self.dim_in_node = dim_in_node

        self.src_fc = jittor.nn.Linear(dim_in_time + dim_in_node, 100)
        self.dst_fc = jittor.nn.Linear(dim_in_time + dim_in_node, 100)
        self.out_fc = jittor.nn.Linear(100, 1)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        # self.src_fc.reset_parameters()
        # self.dst_fc.reset_parameters()
        # self.out_fc.reset_parameters()
        reset_linear_parameters(self.src_fc)
        reset_linear_parameters(self.dst_fc)
        reset_linear_parameters(self.out_fc)

    def execute(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = nn.relu(h_src + h_pos_dst)
        h_neg_edge = nn.relu(h_src.repeat(neg_samples, 1) + h_neg_dst)
        # h_pos_edge = jittor.nn.functional.relu(h_pos_dst)
        # h_neg_edge = jittor.nn.functional.relu(h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
    
class Mixer_per_node(nn.Module):
    """
    Wrapper of MLPMixer and EdgePredictor
    """
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(Mixer_per_node, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']

        if self.time_feats_dim > 0:
            self.base_model = MLPMixer(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        
        self.creterion = nn.BCEWithLogitsLoss()
        self.reset_parameters()            

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        
    def execute(self, model_inputs, has_temporal_neighbors, neg_samples, node_feats):        
        pred_pos, pred_neg = self.predict(model_inputs, has_temporal_neighbors, neg_samples, node_feats)
        
        pos_mask, neg_mask = self.pos_neg_mask(has_temporal_neighbors, neg_samples)
        # loss_pos = self.creterion(pred_pos, jittor.ones_like(pred_pos))[pos_mask].mean()
        # loss_neg = self.creterion(pred_neg, jittor.zeros_like(pred_neg))[neg_mask].mean()
        loss_pos = self.creterion(pred_pos, jittor.ones_like(pred_pos))[jittor.array(pos_mask)].mean()
        loss_neg = self.creterion(pred_neg, jittor.zeros_like(pred_neg))[jittor.array(neg_mask)].mean()
        
        # compute roc and precision score
        acc, auc  = compute_ap_score(pred_pos, pred_neg, neg_samples)        
        return loss_pos + loss_neg, acc, auc
    
    def predict(self, model_inputs, has_temporal_neighbors, neg_samples, node_feats):
        
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = jittor.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            print('Either time_feats_dim or node_feats_dim must larger than 0!')
        
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg
    
    def pos_neg_mask(self, mask, neg_samples):
        num_edge = len(mask) // (neg_samples + 2)
        src_mask = mask[:num_edge]
        pos_dst_mask = mask[num_edge:2 * num_edge]
        neg_dst_mask = mask[2 * num_edge:]

        pos_mask = [(i and j) for i,j in zip(src_mask, pos_dst_mask)]
        neg_mask = [(i and j) for i,j in zip(src_mask * neg_samples, neg_dst_mask)]
        return pos_mask, neg_mask
    
    


################################################################################################
################################################################################################
################################################################################################

"""
Module: Node classifier
"""


class NodeClassificationModel(nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = jittor.nn.Linear(dim_in, dim_hid)
        self.fc2 = jittor.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        return x