import numpy as np
import torch
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from e3nn.math import soft_one_hot_linspace

@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_node_output : `e3nn.o3.Irreps` or None
        representation of the output node features

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        number of nodes convolved over calculated on the fly
    """

    def __init__(
        self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(fc_neurons + [tp.weight_numel], torch.nn.functional.silu)
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, num_neighbors) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out
    

def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class MessagePassing(torch.nn.Module):
    r"""

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input features

    irreps_node_hidden : `e3nn.o3.Irreps`
        representation of the hidden features

    irreps_node_output : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the nodes attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    layers : int
        number of gates (non linearities)

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_hidden,
        irreps_node_output,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        fc_neurons,
    ) -> None:
        super().__init__()

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        irreps_node = self.irreps_node_input

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps_node, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in, fc_neurons
            )
            irreps_node = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps_node, self.irreps_node_attr, self.irreps_edge_attr, self.irreps_node_output, fc_neurons
            )
        )

    def forward(self, node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        for lay in self.layers:
            num_neighbors = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=node_features.shape[0]).unsqueeze(-1) 
            node_features = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, num_neighbors)

        return node_features
    
class E3NN(torch.nn.Module):
    def __init__(
        self,
        node_dim=8, # multiples of 4 
        out_dim=4,
        softmax=True,
        irreps_node_output='1x1o', # one type-1 odd feature
        rbf_cutoff=1.0,
        n_rbf=8,
        smooth_cutoff=True,
        mul=16,
        layers=3,
        lmax=2,
        pool_nodes=True,
        normalize=True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.rbf_cutoff = rbf_cutoff
        self.smooth_cutoff = smooth_cutoff 
        self.node_dim = node_dim
        self.n_rbf = n_rbf

        self.irreps_node_input = o3.Irreps([(self.node_dim//4, (0, 1)),(self.node_dim//4,(1,1))]) #8x0e
        self.node_embedding = nn.Embedding(100, self.node_dim)
        self.neighbor_embed = nn.Sequential(nn.Linear(self.node_dim, self.node_dim*2),
                                            nn.ReLU(),
                                            nn.Linear(self.node_dim*2, self.node_dim))
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        if out_dim is not None:
            self.out_proj = nn.Linear(self.irreps_node_output.dim, out_dim)
        else:
            self.out_proj = nn.Identity()

        if softmax:
            self.softmax = nn.Softmax(dim=1)
        else:
            self.softmax = nn.Identity()

        self.pool_nodes = pool_nodes
        self.normalize = normalize

        irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        ) # mul x (l+1)^2

        self.sph = o3.SphericalHarmonics(
            list(range(1,self.lmax + 1)), normalize=self.normalize, normalization="component"
        )
        self.mp = MessagePassing(
            irreps_node_input=self.irreps_node_input,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=self.irreps_node_output,
            irreps_node_attr=self.irreps_node_input,
            irreps_edge_attr=self.sph.irreps_out, 
            layers=layers,
            fc_neurons=[self.n_rbf, 2*self.n_rbf]
        )

    def forward(self, inputs) -> torch.Tensor:
        atomic_numbers, batch, edge_index, edge_weight, edge_vec = inputs.z, inputs.batch, inputs.edge_index, inputs.edge_weight, inputs.edge_vec
        device = inputs.z.device
        if batch is None:
            batch = torch.zeros(atomic_numbers.shape[0], dtype=torch.long, device=device)
        self.num_nodes = torch.sum(batch == 0) # N

        node_attr = self.node_embedding(atomic_numbers).to(device)

        edge_src = edge_index[0] # E,1
        edge_dst = edge_index[1] # E,1
        edge_vec = F.normalize(edge_vec,dim=-1)
        edge_vec = self.sph(edge_vec) # E, (l+1)^2-1
        edge_weight = soft_one_hot_linspace(
            edge_weight,
            0.0,
            self.rbf_cutoff,
            self.n_rbf,
            basis="cosine",  
            cutoff=self.smooth_cutoff,  
        ).mul(self.n_rbf**0.5) # num rbf
        node_attr = self.mp(node_attr, node_attr, edge_src, edge_dst, edge_vec, edge_weight)
   
        if self.pool_nodes:
            node_attr = scatter(node_attr, batch, dim=0, reduce="sum").div(self.num_nodes**0.5)
            node_attr = F.normalize(node_attr,dim=-1)
            return self.softmax(self.out_proj(node_attr)), node_attr
        else:
            return node_attr

