
__all__ = ['Transpose', 'get_activation_fn', 'get_norm_layer', 'PositionalEncoding', 'SinCosPosEncoding',
           'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding', 'TransposedInstanceNorm1d',
           'fc_layer', 'fc_layer_split',
           'ConvLayers', 'DeconvLayers', 'Flatten', 'MLP', 'Reshape',
           'log_Normal_standard', 'log_Normal_diag', 'weighted_average']

import torch
from torch import nn
from torch.nn.parameter import Parameter
import math
from models.normalization.batchinstancenorm import BatchInstanceNorm1d
from models.normalization.switchable_norm import SwitchNorm1d
from models.normalization.stoch_norm import StochNorm1d
import numpy as np


class TransposedLayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.layer = nn.Sequential(Transpose(1, 2), nn.LayerNorm(*args, **kwargs), Transpose(1, 2))

    def forward(self, x):
        # x is (N, C, L)
        return self.layer(x)


class InstanceNorm1d_affined(nn.Module):
    def __init__(self, num_features, *args, **kwargs):
        super().__init__()

        self.layer = nn.InstanceNorm1d(num_features, affine=True, *args, **kwargs)

    def forward(self, x):
        return self.layer(x)


class TransposedInstanceNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.layer = nn.Sequential(Transpose(1, 2), nn.InstanceNorm1d(*args, **kwargs), Transpose(1, 2))

    def forward(self, x):
        # x is (N, L, C)
        return self.layer(x)


def get_norm_layer(norm):
    """
    input to norm layer: (N, C, L). For LN, transpose is required.
    """
    if norm == 'BN':
        NormLayer = nn.BatchNorm1d
    elif norm == 'LN':
        NormLayer = TransposedLayerNorm  # Qz: Sequential of [Transpose, LN, Transpose]
    elif norm == 'IN':
        # NormLayer = nn.InstanceNorm1d  # Without Learnable Parameters
        NormLayer = InstanceNorm1d_affined  # With Learnable Parameters
    elif norm == 'BIN':
        NormLayer = BatchInstanceNorm1d
    elif norm == 'SwitchNorm':
        NormLayer = SwitchNorm1d
    elif norm == 'StochNorm':  # Only 2D now
        NormLayer = StochNorm1d
    else:
        raise ValueError("No such normalization type")
    return NormLayer


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (
                    torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        # pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


####################### VAE related ############################
""" Layers """
class Identity(nn.Module):
    '''A nn-module to simply pass on the input data.'''
    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''
    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.reshape(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Reshape(nn.Module):
    '''A nn-module to reshape a tensor(-tuple) to a 3-dim "MTS"-tensor(-tuple) with [ts_channels] channels.'''
    def __init__(self, ts_channels):
        super().__init__()
        self.ts_channels = ts_channels

    def forward(self, x):
        if type(x)==tuple:
            batch_size = x[0].size(0)   # first dimenstion should be batch-dimension.
            ts_length = int(x[0].nelement() / (batch_size * self.ts_channels))
            return (x_item.reshape(batch_size, self.ts_channels, ts_length) for x_item in x)
        else:
            batch_size = x.size(0)   # first dimenstion should be batch-dimension.
            ts_length = int(x.nelement() / (batch_size * self.ts_channels))
            return x.reshape(batch_size, self.ts_channels, ts_length)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.ts_channels)
        return tmpstr


def linearExcitability(input, weight, excitability=None, bias=None):
    '''Applies a linear transformation to the incoming data: :math:`y = c(xA^T) + b`.

    Shape:
        - input:        :math:`(N, *, in_features)`
        - weight:       :math:`(out_features, in_features)`
        - excitability: :math:`(out_features)`
        - bias:         :math:`(out_features)`
        - output:       :math:`(N, *, out_features)`
    (NOTE: `*` means any number of additional dimensions)'''

    if excitability is not None:
        output = input.matmul(weight.t()) * excitability
    else:
        output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output


class LinearExcitability(nn.Module):
    '''Module for a linear transformation with multiplicative excitability-parameter (i.e., learnable) and/or -buffer.

    Args:
        in_features:    size of each input sample
        out_features:   size of each output sample
        bias:           if 'False', layer will not learn an additive bias-parameter (DEFAULT=True)
        excitability:   if 'True', layer will learn a multiplicative excitability-parameter (DEFAULT=False)
        excit_buffer:   if 'True', layer will have excitability-buffer whose value can be set (DEFAULT=False)

    Shape:
        - input:    :math:`(N, *, in_features)` where `*` means any number of additional dimensions
        - output:   :math:`(N, *, out_features)` where all but the last dimension are the same shape as the input.

    Attributes:
        weight:         the learnable weights of the module of shape (out_features x in_features)
        excitability:   the learnable multiplication terms (out_features)
        bias:           the learnable bias of the module of shape (out_features)
        excit_buffer:   fixed multiplication variable (out_features)'''

    def __init__(self, in_features, out_features, bias=True, excitability=False, excit_buffer=False):
        super(LinearExcitability, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if excitability:
            self.excitability = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('excitability', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if excit_buffer:
            buffer = torch.Tensor(out_features).uniform_(1,1)
            self.register_buffer("excit_buffer", buffer)
        else:
            self.register_buffer("excit_buffer", None)
        self.reset_parameters()

    def reset_parameters(self):
        '''Modifies the parameters "in-place" to initialize / reset them at appropriate values.'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.excitability is not None:
            self.excitability.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        '''Running this model's forward step requires/returns:
            -[input]:   [batch_size]x[...]x[in_features]
            -[output]:  [batch_size]x[...]x[hidden_features]'''
        if self.excit_buffer is None:
            excitability = self.excitability
        elif self.excitability is None:
            excitability = self.excit_buffer
        else:
            excitability = self.excitability*self.excit_buffer
        return linearExcitability(input, self.weight, excitability, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'

class fc_layer(nn.Module):
    '''Fully connected layer, with possibility of returning "pre-activations".

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, out_size, nl=nn.ReLU(), drop=0., bias=True, batch_norm=False,
                 excitability=False, excit_buffer=False, gated=False, phantom=False):
        super().__init__()
        self.bias = False if batch_norm else bias
        if drop>0:
            self.dropout = nn.Dropout(drop)
        self.linear = LinearExcitability(in_size, out_size, bias=False if batch_norm else bias,
                                            excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gated:
            self.gate = nn.Linear(in_size, out_size)
            self.sigmoid = nn.Sigmoid()
        if phantom:
            self.phantom = nn.Parameter(torch.zeros(out_size), requires_grad=True)
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl=="none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else Identity())

    def forward(self, x, return_pa=False, **kwargs):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        if hasattr(self, 'phantom'):
            gated_pre_activ = gated_pre_activ + self.phantom
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]


class fc_layer_split(nn.Module):
    '''Fully connected layer outputting [mean] and [logvar] for each unit.

    Input:  [batch_size] x ... x [in_size] tensor
    Output: tuple with two [batch_size] x ... x [out_size] tensors'''

    def __init__(self, in_size, out_size, nl_mean=nn.Sigmoid(), nl_logvar=nn.Hardtanh(min_val=-4.5, max_val=0.),
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False, gated=False):
        super().__init__()

        self.mean = fc_layer(in_size, out_size, drop=drop, bias=bias, excitability=excitability,
                             excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_mean)
        self.logvar = fc_layer(in_size, out_size, drop=drop, bias=False, excitability=excitability,
                               excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_logvar)

    def forward(self, x):
        return (self.mean(x), self.logvar(x))

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list


class MLP(nn.Module):
    '''Module for a multi-layer perceptron (MLP).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor'''

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None, size_per_layer=None,
                 drop=0, batch_norm=False, nl="relu", bias=True, excitability=False, excit_buffer=False, gated=False,
                 phantom=False, output='normal'):
        '''sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [gated]            <bool>; if True, each linear layer has an additional learnable gate
                                    (whereby the gate is controlled by the same input as that goes through the gate)
        [phantom]          <bool>; if True, add phantom parameters to pre-activations, used for computing KFAC Fisher
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "none", final layer has no non-linearity
                                     - "sigmoid", final layer has sigmoid non-linearity'''

        super().__init__()
        self.output = output

        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if (hid_smooth is not None):
                    hidden_sizes = [int(x) for x in np.linspace(hid_size, hid_smooth, num=layers-1)]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size] if layers>0 else [input_size]
        self.layers = len(size_per_layer)-1

        # set label for this module
        # -determine "non-default options"-label
        nd_label = "{drop}{bias}{exc}{bn}{nl}{gate}".format(
            drop="" if drop==0 else "d{}".format(drop),
            bias="" if bias else "n", exc="e" if excitability else "", bn="b" if batch_norm else "",
            nl="l" if nl=="leakyrelu" else ("n" if nl=="none" else ""), gate="g" if gated else "",
        )
        nd_label = "{}{}".format("" if nd_label=="" else "-{}".format(nd_label),
                                 "" if output=="normal" else "-{}".format(output))
        # -set label
        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement=="" else "x", i)
        self.label = "F{}{}".format(size_statement, nd_label) if self.layers>0 else ""

        # set layers
        for lay_id in range(1, self.layers+1):
            # number of units of this layer's input and output
            in_size = size_per_layer[lay_id-1]
            out_size = size_per_layer[lay_id]
            # define and set the fully connected layer
            layer = fc_layer(
                in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer,
                batch_norm=False if (lay_id==self.layers and not output=="normal") else batch_norm, gated=gated,
                nl=("none" if output=="none" else nn.Sigmoid()) if (
                    lay_id==self.layers and not output=="normal"
                ) else nl, drop=drop if lay_id>1 else 0., phantom=phantom
            )
            setattr(self, 'fcLayer{}'.format(lay_id), layer)

        # if no layers, add "identity"-module to indicate in this module's representation nothing happens
        if self.layers<1:
            self.noLayers = Identity()

    def forward(self, x, return_intermediate=False):
        if return_intermediate:
            intermediate = {}
        for lay_id in range(1, self.layers + 1):
            if return_intermediate:
                intermediate[f"fcLayer{lay_id}"] = x
            x = getattr(self, "fcLayer{}".format(lay_id))(x)
        return (x, intermediate) if return_intermediate else x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.layers+1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list


""" Encoder """
class conv_layer(nn.Module):
    '''Standard convolutional layer. Possible to return pre-activations.'''

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                 drop=0, batch_norm=False, nl=nn.ReLU(), bias=True, gated=False):
        super().__init__()
        if drop>0:
            self.dropout = nn.Dropout1d(drop)
        self.conv = nn.Conv1d(in_planes, out_planes, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_planes)
        if gated:
            self.gate = nn.Conv1d(in_planes, out_planes, stride=stride, kernel_size=kernel_size, padding=padding,
                                  bias=False)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl=="none":
            self.nl = nn.ReLU() if nl=="relu" else (nn.LeakyReLU() if nl=="leakyrelu" else Identity())

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.conv(input)) if hasattr(self, 'bn') else self.conv(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.conv]


class ConvLayers(nn.Module):
    '''Convolutional feature extractor model for time series. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x [ts_channels] x [ts_length] tensor
    Output: [batch_size] x [out_channels] x [out_length] tensor
                - with [out_channels] = [start_channels] x 2**[reducing_layers] x [block.expansion]
                       [out_size] = [image_size] / 2**[reducing_layers]'''

    def __init__(self, ts_channels, depth=3, start_channels=16, reducing_layers=None, batch_norm=True, nl="relu",
                 output="normal", global_pooling=False, gated=False):
        '''Initialize stacked convolutional layers (standard).

        [ts_channels]       <int> # channels of input time series to encode
        [depth]             <int> # layers
        [start_channels]    <int> # channels in 1st layer, doubled in every "rl" (=reducing layer)
        [reducing_layers]   <int> # layers in which image-size is halved & # channels doubled (default=[depth]-1)
                                      ("rl"'s are the last conv-layers; in 1st layer # channels cannot double)
        [batch_norm]        <bool> whether to use batch-norm after each convolution-operation
        [nl]                <str> non-linearity to be used: [relu|leakyrelu]
        [output]            <str>  if - "normal", final layer is same as all others
                                      - "none", final layer has no batchnorm or non-linearity
        [global_pooling]    <bool> whether to include global average pooling layer at very end
        [gated]             <bool> whether conv-layers should be gated (not implemented for ResNet-layers)'''

        # Prepare label
        type_label = "C"
        channel_label = "{}-{}x{}".format(ts_channels, depth, start_channels)
        block_label = ""
        nd_label = "{bn}{nl}{gp}{gate}{out}".format(bn="b" if batch_norm else "", nl="l" if nl=="leakyrelu" else "",
                                                    gp="p" if global_pooling else "", gate="g" if gated else "",
                                                    out="n" if output=="none" else "")
        nd_label = "" if nd_label=="" else "-{}".format(nd_label)

        # Set configurations
        super().__init__()
        self.depth = depth

        # Qz: # of reducing layers, each one doubles the # of channels and halves the ts_length
        self.rl = depth-1 if (reducing_layers is None) else (reducing_layers if (depth+1)>reducing_layers else depth)
        rl_label = "" if self.rl==(self.depth-1) else "-rl{}".format(self.rl)
        self.label = "{}{}{}{}{}".format(type_label, channel_label, block_label, rl_label, nd_label)
        self.block_expansion = 1
        # -> constant by which # of output channels of each block is multiplied (if >1, it creates "bottleneck"-effect)
        double_factor = self.rl if self.rl<depth else depth-1 # -> how often # start-channels is doubled
        self.out_channels = (start_channels * 2**double_factor) * self.block_expansion if depth>0 else ts_channels
        # -> number channels in last layer (as seen from image)
        self.start_channels = start_channels  # -> number channels in 1st layer (doubled in every "reducing layer")
        self.global_pooling = global_pooling  # -> whether or not average global pooling layer should be added at end

        # Conv-layers
        output_channels = start_channels
        for layer_id in range(1, depth+1):
            # should this layer down-sample? --> last [self.rl] layers should be down-sample layers
            reducing = True if (layer_id > (depth-self.rl)) else False
            # calculate number of this layer's input and output channels
            input_channels = ts_channels if layer_id==1 else output_channels * self.block_expansion
            output_channels = output_channels*2 if (reducing and not layer_id==1) else output_channels
            # define and set the convolutional-layer
            new_layer = conv_layer(input_channels, output_channels, stride=2 if reducing else 1,
                                   drop=0, nl="no" if output=="none" and layer_id==depth else nl,
                                   batch_norm=False if output=="none" and layer_id==depth else batch_norm,
                                   gated= False if output=="none" and layer_id==depth else gated)

            setattr(self, 'convLayer{}'.format(layer_id), new_layer)
        # Perform pooling (if requested)
        self.pooling = nn.AdaptiveAvgPool1d(1) if global_pooling else Identity()

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False):
        # Initiate <list> for keeping track of intermediate hidden (pre-)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all conv-layers
        for layer_id in range(skip_first+1, self.depth+1-skip_last):
            (x, pre_act) = getattr(self, 'convLayer{}'.format(layer_id))(x, return_pa=True)
            if return_lists:
                pre_act_list.append(pre_act)  #-> for each layer, store pre-activations
                if layer_id<(self.depth-skip_last):
                    hidden_act_list.append(x) #-> for all but last layer, store hidden activations
        # Global average pooling (if requested)
        x = self.pooling(x)
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    def out_size(self, ts_length, ignore_gp=False):
        '''Given [ts_length] of input, return the size of the "final" image that is outputted.'''
        out_size = int(np.ceil(ts_length / 2**(self.rl))) if self.depth>0 else ts_length
        return 1 if (self.global_pooling and not ignore_gp) else out_size

    def out_units(self, ts_length, ignore_gp=False):
        '''Given [ts_length] of input, return the total number of units in the output.'''
        return self.out_channels * self.out_size(ts_length, ignore_gp=ignore_gp)

    def layer_info(self, ts_length):
        '''Return list with shape of all hidden layers.'''
        layer_list = []
        reduce_number = 0  # keep track how often image-size has been halved
        double_number = 0  # keep track how often channel number has been doubled
        for layer_id in range(1, self.depth):
            reducing = True if (layer_id > (self.depth-self.rl)) else False
            if reducing:
                reduce_number += 1
            if reducing and layer_id>1:
                double_number += 1
            pooling = True if self.global_pooling and layer_id==(self.depth-1) else False
            expansion = 1 if layer_id==1 else self.block_expansion
            # add shape of this layer to list
            layer_list.append([(self.start_channels * 2**double_number) * expansion,
                               1 if pooling else int(np.ceil(ts_length / 2**reduce_number))])
        return layer_list

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.depth+1):
            list += getattr(self, 'convLayer{}'.format(layer_id)).list_init_layers()
        return list

    @property
    def name(self):
        return self.label


""" Decoder """
class deconv_layer(nn.Module):
    '''Standard "deconvolutional" layer. Possible to return pre-activations.'''

    def __init__(self, input_channels, output_channels, stride=1, drop=0, batch_norm=True, nl="relu", bias=True,
                 gated=False, smaller_kernel=False):
        super().__init__()
        if drop>0:
            self.dropout = nn.Dropout1d(drop)
        self.deconv = nn.ConvTranspose1d(input_channels, output_channels, bias=bias, stride=stride,
                                         kernel_size=(2 if smaller_kernel else 4) if stride==2 else 3,
                                         padding=0 if (stride==2 and smaller_kernel) else 1)
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_channels)
        if gated:
            self.gate = nn.ConvTranspose1d(input_channels, output_channels, bias=False, stride=stride,
                                           kernel_size=(2 if smaller_kernel else 4) if stride==2 else 3,
                                           padding=0 if (stride==2 and smaller_kernel) else 1)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif nl in ("sigmoid", "hardtanh"):
            self.nl = nn.Sigmoid() if nl=="sigmoid" else nn.Hardtanh(min_val=-4.5, max_val=0)
        elif not nl=="none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else Identity())

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.deconv(input)) if hasattr(self, 'bn') else self.deconv(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.deconv]


class DeconvLayers(nn.Module):
    '''"Deconvolutional" feature decoder model for time series. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x [in_channels] x [in_length] tensor
    Output: (tuple of) [batch_size] x [ts_channels] x [final_length] tensor
                - with [final_size] = [in_size] x 2**[reducing_layers]
                       [in_channels] = [final_channels] x 2**min([reducing_layers], [depth]-1)'''

    def __init__(self, ts_channels, final_channels=16, depth=3, reducing_layers=None, batch_norm=True, nl="relu",
                 gated=False, output="normal", smaller_kernel=False,):
        '''
        [ts_channels]       # channels of time series to decode
        [final_channels]    # channels in layer before output, was halved in every "rl" (=reducing layer) when moving
                                through model; corresponds to [start_channels] in "ConvLayers"-module
        [depth]             # layers (seen from the image, # channels is halved in each layer going to output image)
        [reducing_layers]   # of layers in which image-size is doubled & number of channels halved (default=[depth]-1)
                               ("rl"'s are the first conv-layers encountered--i.e., last conv-layers as seen from image)
                               (note that in the last layer # channels cannot be halved)
        [batch_norm]        <bool> whether to use batch-norm after each convolution-operation
        [nl]                <str> what non-linearity to use -- choices: [relu, leakyrelu, sigmoid, none]
        [gated]             <bool> whether deconv-layers should be gated
        [output]            <str>; if - "normal", final layer is same as all others
                                      - "none", final layer has no non-linearity
                                      - "sigmoid", final layer has sigmoid non-linearity
        [smaller_kernel]    <bool> if True, use kernel-size of 2 (instead of 4) & without padding in reducing-layers'''

        # configurations
        super().__init__()
        self.depth = depth if depth>0 else 0
        self.rl = self.depth-1 if (reducing_layers is None) else min(self.depth, reducing_layers)
        type_label = "Deconv"
        nd_label = "{bn}{nl}{gate}{out}".format(bn="-bn" if batch_norm else "", nl="-lr" if nl=="leakyrelu" else "",
                                                gate="-gated" if gated else "",
                                                out="" if output=="normal" else "-{}".format(output))
        self.label = "{}-ic{}-{}x{}-rl{}{}{}".format(type_label, ts_channels, self.depth, final_channels, self.rl,
                                                     "s" if smaller_kernel else "", nd_label)
        if self.depth>0:
            self.in_channels = final_channels * 2**min(self.rl, self.depth-1) # -> input-channels for deconv
            self.final_channels = final_channels                              # -> channels in layer before output
        self.ts_channels = ts_channels                                  # -> output-channels for deconv

        # "Deconv"- / "transposed conv"-layers
        if self.depth>0:
            output_channels = self.in_channels
            for layer_id in range(1, self.depth+1):
                # should this layer down-sample? --> first [self.rl] layers should be down-sample layers
                reducing = True if (layer_id<(self.rl+1)) else False
                # update number of this layer's input and output channels
                input_channels = output_channels
                output_channels = int(output_channels/2) if reducing else output_channels
                # define and set the "deconvolutional"-layer
                new_layer = deconv_layer(
                    input_channels, output_channels if layer_id<self.depth else ts_channels,
                    stride=2 if reducing else 1, batch_norm=batch_norm if layer_id<self.depth else False,
                    nl=nl if layer_id<self.depth or output=="normal" else (
                        "none" if output=="none" else nn.Sigmoid()
                    ), gated=gated, smaller_kernel=smaller_kernel
                )

                setattr(self, 'deconvLayer{}'.format(layer_id), new_layer)

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False):
        # Initiate <list> for keeping track of intermediate hidden (pre-)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all "deconv"-layers
        if self.depth>0:
            for layer_id in range(skip_first+1, self.depth+1-skip_last):
                (x, pre_act) = getattr(self, 'deconvLayer{}'.format(layer_id))(x, return_pa=True)
                if return_lists:
                    pre_act_list.append(pre_act)  #-> for each layer, store pre-activations
                    if layer_id<(self.depth-skip_last):
                        hidden_act_list.append(x) #-> for all but last layer, store hidden activations
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    def ts_length(self, in_units):
        '''Given the number of units fed in, return the length of the target ts.'''
        if self.depth>0:
            input_ts_length = in_units/self.in_channels #-> length of ts fed to last layer (seen from image)
            return input_ts_length * 2**self.rl
        else:
            return in_units / self.ts_channels

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.depth+1):
            list += getattr(self, 'deconvLayer{}'.format(layer_id)).list_init_layers()
        return list

    @property
    def name(self):
        return self.label


########################################################
## Calculate log-likelihood for various distributions ##
########################################################

def log_Normal_standard(x, mean=0, average=False, dim=None):
    '''Calculate log-likelihood of sample [x] under Gaussian distribution(s) with mu=[mean], diag_var=I.
    NOTES: [dim]=-1    summing / averaging over all but the first dimension
           [dim]=None  summing / averaging is done over all dimensions'''
    log_normal = -0.5 * torch.pow(x-mean, 2)
    if dim is not None and dim == -1:
        log_normal = log_normal.reshape(log_normal.size(0), -1)
        dim = 1
    if average:
        return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)

def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    '''Calculate log-likelihood of sample [x] under Gaussian distribution(s) with mu=[mean], diag_var=exp[log_var].
    NOTES: [dim]=-1    summing / averaging over all but the first dimension
           [dim]=None  summing / averaging is done over all dimensions'''
    log_normal = -0.5 * (log_var + torch.pow(x-mean, 2) / torch.exp(log_var))
    if dim is not None and dim==-1:
        log_normal = log_normal.reshape(log_normal.size(0), -1)
        dim = 1
    if average:
        return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)


def weighted_average(tensor, weights=None, dim=0):
    '''Computes weighted average of [tensor] over dimension [dim].'''
    if weights is None:
        mean = torch.mean(tensor, dim=dim)
    else:
        batch_size = tensor.size(dim) if len(tensor.size())>0 else 1
        assert len(weights)==batch_size
        #sum_weights = sum(weights)
        #norm_weights = torch.Tensor([weight/sum_weights for weight in weights]).to(tensor.device)
        norm_weights = torch.tensor([weight for weight in weights]).to(tensor.device)
        mean = torch.mean(norm_weights*tensor, dim=dim)
    return mean