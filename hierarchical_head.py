"""
An implementation of the BINN hierarchical inference head from "Structured Label Inference for Visual Understanding"
(https://arxiv.org/pdf/1802.06459.pdf, section 3.1, equations (1)-(4))
"""
import math
import torch
import torch.nn as nn


class Linear2(nn.Module):
    """
    Applies a linear transformation of the form A1 * x1 + A2 * x2 + b
    """

    __constants__ = ["bias", "in_features1", "in_features2", "out_features"]

    def __init__(self, in_features1, in_features2, out_features, bias=True):
        super().__init__()
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features1))
        self.weight2 = nn.Parameter(torch.Tensor(out_features, in_features2))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(
                min(fan_in1, fan_in2)
            )  # use the larger bound for xavier-like initialization
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        output = input1.matmul(self.weight1.t()) + input2.matmul(self.weight2.t())
        if self.bias is not None:
            output += self.bias
        ret = output
        return ret

    def extra_repr(self):
        return "in_features1={}, in_features2={}, out_features={}, bias={}".format(
            self.in_features1,
            self.in_features2,
            self.out_features,
            self.bias is not None,
        )


class HierarchicalHead(nn.Module):
    """
    An implementation of the BINN hierarchical inference head from "Structured Label Inference for Visual Understanding"
    (https://arxiv.org/pdf/1802.06459.pdf)

    The implementation here takes a single feature vector, expected to be the output of a backbone CNN, which is
    linearly transformed into output activations
    """

    def __init__(self, input_feature_length, layer_nodes):
        """
        :param input_feature_dim: The length of the input feature vector, referred to in the paper as D. Should be an
        int.
        :param layer_nodes: A tuple or list with length equal to num_layers, each of which is referred to in the paper
        as n^l. Should be a tuple or list. The length of layer_nodes is the number of layers, referred to in the paper
        as m.
        """
        super().__init__()

        # # check layer_nodes is formatted correctly and has more than one element
        # assert isinstance(
        #     layer_nodes, (list, tuple)
        # ), f"layer_nodes parameter ({layer_nodes}) must be a tuple or list."
        # assert (
        #     len(layer_nodes) > 1
        # ), f"Number of layer nodes ({layer_nodes}) must be greater than one."

        self.D = input_feature_length
        self.m = len(layer_nodes)
        self.n = layer_nodes

        # Create the list of input transforms (equation (1) of the paper), which are FC layers of shape (D, n^l)
        input_transforms = []
        for n in self.n:
            input_transforms.append(nn.Linear(self.D, n))
        self.input_transforms = nn.ModuleList(input_transforms)

        # Create the list of forward and backward linear layers (equations (2)-(3) of the paper, the first of which is a
        # one-to-one FC layer of shape (n^l, n^l), the rest of which are two-to-one FC layers of shape (n^l-1, n^l, n^l)
        # or (n^l+1, n^l, n^l)
        forward_transforms = [nn.Linear(self.n[0], self.n[0])]
        for i in range(1, self.m):
            forward_transforms.append(Linear2(self.n[i - 1], self.n[i], self.n[i]))
        self.forward_transforms = nn.ModuleList(forward_transforms)

        backward_transforms = [nn.Linear(self.n[self.m - 1], self.n[self.m - 1])]
        for i in range(self.m - 2, -1, -1):
            backward_transforms.append(Linear2(self.n[i + 1], self.n[i], self.n[i]))
        self.backward_transforms = nn.ModuleList(backward_transforms)

        # Create the list of output transforms, which are two-to-one FC layers of shape (n^l, n^l, n^l)
        output_transforms = []
        for i in self.n:
            output_transforms.append(Linear2(i, i, i))
        self.output_transforms = nn.ModuleList(output_transforms)

    def forward(self, x):
        """
        Expects x to be a single-dimensional output feature vector (not counting the batch dimension) from the backbone
        CNN, and to have length equal to the input_feature_length passed to __init__

        :param x: A tensor of size (Batch, input_feature_length

        :returns: A list of output logits, which have order and lengths equal to self.layer_nodes
        """

        # stage 1: build the feature vectors (x^l) for each inference layer
        x_l = []
        for net in self.input_transforms:
            x_l.append(net(x))

        # stage 2: build the forward and backward activations (a^l_forward and a^l_backward)
        forward_activations = [self.forward_transforms[0](x_l[0])]
        for i, net in enumerate(self.forward_transforms[1:]):
            forward_activations.append(net(forward_activations[i], x_l[i + 1]))

        backward_activations = [self.backward_transforms[0](x_l[-1])]
        for i, net in enumerate(self.backward_transforms[1:]):
            backward_activations.append(net(backward_activations[i], x_l[-i - 2]))

        # reverse backward activations for easier indexing (should do this with reversed() for less memory)
        backward_activations = reversed(backward_activations)

        # stage 3: build output logits
        output_logits = []
        for net, forward_activation, backward_activation in zip(
            self.output_transforms, forward_activations, backward_activations
        ):
            output_logits.append(net(forward_activation, backward_activation))

        return output_logits