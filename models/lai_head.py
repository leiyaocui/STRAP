import math
import torch
import torch.nn as nn


class LinearDual(nn.Module):
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
            )
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


class LAI(nn.Module):

    def __init__(self, length, layer_nodes):
        """
        :param input_feature_dim: The length of the flatten feature without batchsize.
        :param layer_nodes: A tuple or a list which includes the number of object concepts and affordance concepts. For examples, (12, 6)
        """
        super().__init__()
        self.D = length
        self.m = len(layer_nodes)
        self.n = layer_nodes

        input_transforms = []
        for n in self.n:
            input_transforms.append(nn.Linear(self.D, n))
        self.input_transforms = nn.ModuleList(input_transforms)

        forward_transforms = [nn.Linear(self.n[0], self.n[0])]
        for i in range(1, self.m):
            forward_transforms.append(LinearDual(self.n[i - 1], self.n[i], self.n[i]))
        self.forward_transforms = nn.ModuleList(forward_transforms)

        backward_transforms = [nn.Linear(self.n[self.m - 1], self.n[self.m - 1])]
        for i in range(self.m - 2, -1, -1):
            backward_transforms.append(LinearDual(self.n[i + 1], self.n[i], self.n[i]))
        self.backward_transforms = nn.ModuleList(backward_transforms)

        output_transforms = []
        for i in self.n:
            output_transforms.append(LinearDual(i, i, i))
        self.output_transforms = nn.ModuleList(output_transforms)

    def forward(self, x):
        x_l = []
        for net in self.input_transforms:
            x_l.append(net(x))

        forward_activations = [self.forward_transforms[0](x_l[0])]
        for i, net in enumerate(self.forward_transforms[1:]):
            forward_activations.append(net(forward_activations[i], x_l[i + 1]))

        backward_activations = [self.backward_transforms[0](x_l[-1])]
        for i, net in enumerate(self.backward_transforms[1:]):
            backward_activations.append(net(backward_activations[i], x_l[-i - 2]))

        backward_activations = reversed(backward_activations)

        output_logits = []
        for net, forward_activation, backward_activation in zip(
            self.output_transforms, forward_activations, backward_activations
        ):
            output_logits.append(net(forward_activation, backward_activation))

        return output_logits