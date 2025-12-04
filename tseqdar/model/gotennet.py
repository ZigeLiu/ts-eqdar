from functools import partial
from typing import Callable, Optional, Tuple, Mapping, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter, softmax
from torch_geometric.nn.pool import global_mean_pool, global_max_pool

from tseqdar.utils.components import Dense, str2basis, get_weight_init_by_string, str2act, MLP, CosineCutoff, \
    TensorLayerNorm
from tseqdar.utils.components import parse_update_info, TensorInit, NodeInit, EdgeInit
from tseqdar.utils.components import shifted_softplus
from tseqdar.utils.outputs import GatedEquivariantBlock
from tseqdar.utils.utils import (
    RankedLogger,
)

log = RankedLogger(__name__, rank_zero_only=True)

# num_nodes and hidden_dims are placeholder values, will be overwritten by actual data
num_nodes = hidden_dims = 1


def lmax_tensor_size(lmax):
    return ((lmax + 1) ** 2) - 1


def split_degree(tensor, lmax, dim=-1):  # default to last dim
    cumsum = 0
    tensors = []
    for i in range(1, lmax + 1):
        count = lmax_tensor_size(i) - lmax_tensor_size(i - 1)
        # Create slice object for the specified dimension
        slc = [slice(None)] * tensor.ndim  # Create list of slice(None) for all dims
        slc[dim] = slice(cumsum, cumsum + count)  # Replace desired dim with actual slice
        tensors.append(tensor[tuple(slc)])
        cumsum += count
    return tensors


class GATA(MessagePassing):

    def __init__(self, n_atom_basis: int, activation: Callable, weight_init=nn.init.xavier_uniform_,
                 bias_init=nn.init.zeros_,
                 aggr="add", node_dim=0, epsilon: float = 1e-7,
                 layer_norm="", vector_norm="", cutoff=None, num_heads=8, dropout=0.0,
                 edge_updates=True, last_layer=False, scale_edge=True,
                 edge_ln='', evec_dim=None, emlp_dim=None, sep_vecj=True, lmax=1):
        """
        Args:
            n_atom_basis (int): Number of features to describe atomic environments.
            activation (Callable): Activation function to be used. If None, no activation function is used.
            weight_init (Callable): Weight initialization function.
            bias_init (Callable): Bias initialization function.
            aggr (str): Aggregation method ('add', 'mean' or 'max').
            node_dim (int): The axis along which to aggregate.
        """
        super(GATA, self).__init__(aggr=aggr, node_dim=node_dim)
        self.lmax = lmax
        self.sep_vecj = sep_vecj
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.edge_updates = edge_updates
        self.scale_edge = scale_edge
        self.activation = activation

        self.update_info = parse_update_info(edge_updates)

        self.dropout = dropout
        self.n_atom_basis = n_atom_basis

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)
        self.gamma_s = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

        self.num_heads = num_heads
        self.q_w = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.k_w = InitDense(n_atom_basis, n_atom_basis, activation=None)

        self.gamma_v = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

        self.phik_w_ra = InitDense(
            n_atom_basis,
            n_atom_basis,
            activation=activation,
        )

        InitMLP = partial(MLP, weight_init=weight_init, bias_init=bias_init)

        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim
        if not self.last_layer and self.edge_updates:
            if self.update_info["mlp"] or self.update_info["mlpa"]:
                dims = [n_atom_basis, self.edge_mlp_dim, n_atom_basis]
            else:
                dims = [n_atom_basis, n_atom_basis]
            self.edge_attr_up = InitMLP(dims, activation=activation,
                                        last_activation=None if self.update_info["mlp"] else self.activation,
                                        norm=edge_ln)
            self.vecq_w = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.sep_vecj:
                self.veck_w = nn.ModuleList([
                    InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)
                    for i in range(self.lmax)
                ])
            else:
                self.veck_w = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.update_info["lin_w"] > 0:
                modules = []
                if self.update_info["lin_w"] % 10 == 2:
                    modules.append(self.activation)
                self.lin_w_linear = InitDense(self.edge_vec_dim, n_atom_basis, activation=None,
                                              norm="layer" if self.update_info["lin_ln"] == 2 else "")
                modules.append(self.lin_w_linear)
                self.lin_w = nn.Sequential(*modules)

        self.down_proj = nn.Identity()

        if cutoff is not None:
            self.cutoff = CosineCutoff(cutoff)
        else:
            self.cutoff = nn.Identity()

        self._alpha = None

        self.w_re = InitDense(
            n_atom_basis,
            n_atom_basis * 3,
            None,
        )

        self.layernorm_ = layer_norm
        self.vector_norm_ = vector_norm
        if layer_norm != "":
            self.layernorm = nn.LayerNorm(n_atom_basis)
        else:
            self.layernorm = nn.Identity()
        if vector_norm != "":
            self.tln = TensorLayerNorm(n_atom_basis, trainable=False, norm_type=vector_norm)
        else:
            self.tln = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        if self.layernorm_:
            self.layernorm.reset_parameters()
        if self.vector_norm_:
            self.tln.reset_parameters()
        for l in self.gamma_s:
            l.reset_parameters()

        self.q_w.reset_parameters()
        self.k_w.reset_parameters()
        for l in self.gamma_v:
            l.reset_parameters()
        # self.v_w.reset_parameters()
        # self.out_w.reset_parameters()
        self.w_re.reset_parameters()

        if not self.last_layer and self.edge_updates:
            self.edge_attr_up.reset_parameters()
            self.vecq_w.reset_parameters()

            if self.sep_vecj:
                for w in self.veck_w:
                    w.reset_parameters()
            else:
                self.veck_w.reset_parameters()

            if self.update_info["lin_w"] > 0:
                self.lin_w_linear.reset_parameters()

    def forward(
            self,
            edge_index,
            s: torch.Tensor,
            t: torch.Tensor,
            dir_ij: torch.Tensor,
            r_ij: torch.Tensor,
            d_ij: torch.Tensor,
            num_edges_expanded: torch.Tensor,
    ):
        """Compute interaction output. """
        s = self.layernorm(s)
        t = self.tln(t)

        q = self.q_w(s).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.k_w(s).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        x = self.gamma_s(s)
        val = self.gamma_v(s)
        f_ij = r_ij
        r_ij_attn = self.phik_w_ra(r_ij)
        r_ij = self.w_re(r_ij)

        # propagate_type: (x: Tensor, ten: Tensor, q:Tensor, k:Tensor, val:Tensor, r_ij: Tensor, r_ij_attn: Tensor, d_ij:Tensor, dir_ij: Tensor, num_edges_expanded: Tensor)
        su, tu = self.propagate(edge_index=edge_index, x=x, q=q, k=k, val=val,
                                ten=t, r_ij=r_ij, r_ij_attn=r_ij_attn, d_ij=d_ij, dir_ij=dir_ij,
                                num_edges_expanded=num_edges_expanded)  # , f_ij=f_ij

        s = s + su
        t = t + tu

        if not self.last_layer and self.edge_updates:
            vec = t

            w1 = self.vecq_w(vec)
            if self.sep_vecj:
                vec_split = split_degree(vec, self.lmax, dim=1)
                w_out = torch.concat([
                    w(vec_split[i]) for i, w in enumerate(self.veck_w)
                ], dim=1)

            else:
                w_out = self.veck_w(vec)

            # edge_updater_type: (w1: Tensor, w2:Tensor,  d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, w1=w1, w2=w_out, d_ij=dir_ij, f_ij=f_ij)
            df_ij = f_ij + df_ij
            #self._alpha = None
            return s, t, df_ij
        else:
            #self._alpha = None
            return s, t, f_ij

        return s, t

    def message(
            self,
            edge_index,
            x_i: torch.Tensor,
            x_j: torch.Tensor,
            q_i: torch.Tensor,
            k_j: torch.Tensor,
            val_j: torch.Tensor,
            ten_j: torch.Tensor,
            r_ij: torch.Tensor,
            r_ij_attn: torch.Tensor,
            d_ij: torch.Tensor,
            dir_ij: torch.Tensor,
            num_edges_expanded: torch.Tensor,
            index: torch.Tensor, ptr: OptTensor,
            dim_size: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute message passing.
        """

        r_ij_attn = r_ij_attn.reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        attn = (q_i * k_j * r_ij_attn).sum(dim=-1, keepdim=True)

        attn = softmax(attn, index, ptr, dim_size)

        # Normalize the attention scores
        if self.scale_edge:
            norm = torch.sqrt(num_edges_expanded.reshape(-1, 1, 1)) / np.sqrt(self.n_atom_basis)
        else:
            norm = 1.0 / np.sqrt(self.n_atom_basis)
        attn = attn * norm
        self._alpha = attn
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        self_attn = attn * val_j.reshape(-1, self.num_heads, (self.n_atom_basis * 3) // self.num_heads)
        SEA = self_attn.reshape(-1, 1, self.n_atom_basis * 3)

        x = SEA + (r_ij.unsqueeze(1) * x_j * self.cutoff(d_ij.unsqueeze(-1).unsqueeze(-1)))

        o_s, o_d, o_t = torch.split(x, self.n_atom_basis, dim=-1)
        dmu = o_d * dir_ij[..., None] + o_t * ten_j
        return o_s, dmu

    @staticmethod
    def rej(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def edge_update(self, w1_i, w2_j, w3_j, d_ij, f_ij):
        if self.sep_vecj:
            vi = w1_i
            vj = w2_j
            vi_split = split_degree(vi, self.lmax, dim=1)
            vj_split = split_degree(vj, self.lmax, dim=1)
            d_ij_split = split_degree(d_ij, self.lmax, dim=1)

            pairs = []
            for i in range(len(vi_split)):
                if self.update_info["rej"]:
                    w1 = self.rej(vi_split[i], d_ij_split[i])
                    w2 = self.rej(vj_split[i], -d_ij_split[i])
                    pairs.append((w1, w2))
                else:
                    w1 = vi_split[i]
                    w2 = vj_split[i]
                    pairs.append((w1, w2))
        elif not self.update_info["rej"]:
            w1 = w1_i
            w2 = w2_j
            pairs = [(w1, w2)]
        else:
            w1 = self.rej(w1_i, d_ij)
            w2 = self.rej(w2_j, -d_ij)
            pairs = [(w1, w2)]

        w_dot_sum = None
        for el in pairs:
            w1, w2 = el
            w_dot = (w1 * w2).sum(dim=1)
            if w_dot_sum is None:
                w_dot_sum = w_dot
            else:
                w_dot_sum = w_dot_sum + w_dot
        w_dot = w_dot_sum
        if self.update_info["lin_w"] > 0:
            w_dot = self.lin_w(w_dot)

        if self.update_info["gated"] == "gatedt":
            w_dot = torch.tanh(w_dot)
        elif self.update_info["gated"] == "gated":
            w_dot = torch.sigmoid(w_dot)
        elif self.update_info["gated"] == "act":
            w_dot = self.activation(w_dot)

        df_ij = self.edge_attr_up(f_ij) * w_dot
        return df_ij

    # noinspection PyMethodOverriding
    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, vec

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EQFF(nn.Module):

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8,
                 weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_, vec_dim=None):
        """Equiavariant Feed Forward layer."""
        super(EQFF, self).__init__()
        self.n_atom_basis = n_atom_basis

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        vec_dim = n_atom_basis if vec_dim is None else vec_dim
        context_dim = 2 * n_atom_basis

        self.gamma_m = nn.Sequential(
            InitDense(context_dim, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 2 * n_atom_basis, activation=None),
        )
        self.w_vu = InitDense(
            n_atom_basis, vec_dim, activation=None, bias=False
        )

        self.epsilon = epsilon

    def reset_parameters(self):
        self.w_vu.reset_parameters()
        for l in self.gamma_m:
            l.reset_parameters()

    def forward(self, s, v):
        """Compute Equivariant Feed Forward output."""

        t_prime = self.w_vu(v)
        t_prime_mag = torch.sqrt(torch.sum(t_prime ** 2, dim=-2, keepdim=True) + self.epsilon)
        combined = [s, t_prime_mag]
        combined_tensor = torch.cat(combined, dim=-1)
        m12 = self.gamma_m(combined_tensor)

        m_1, m_2 = torch.split(m12, self.n_atom_basis, dim=-1)
        delta_v = m_2 * t_prime

        s = s + m_1
        v = v + delta_v

        return s, v

from torch_geometric.nn.pool import global_mean_pool
class GotenNet(nn.Module):
    def __init__(
            self,
            out_dim: int = 4, # number of output states
            cg_degree: int=1, # coarse-grain degree, should be consistent with the input cg_index
            softmax = False, # whether to apply softmax to get state model
            n_atom_basis: int = 128,
            n_interactions: int = 8,
            radial_basis: Union[Callable, str] = 'BesselBasis',
            n_rbf: int = 20,
            cutoff = None, # soft cutoff function for edge distances
            rbf_cutoff: float = 5, # rbf cutoff distance for edge distances, should be smaller than cutoff
            activation: Optional[Union[Callable, str]] = F.silu,
            max_z: int = 100,
            epsilon: float = 1e-8,
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            int_layer_norm="",
            int_vector_norm="",
            num_heads=8,
            attn_dropout=0.0,
            edge_updates=True,
            scale_edge=True,
            lmax=2,
            aggr="add",
            edge_ln='',
            evec_dim=None,
            emlp_dim=None,
            sep_int_vec=True,
    ):
        """
        Representation for GotenNet
        """
        super(GotenNet, self).__init__()

        self.scale_edge = scale_edge
        if type(weight_init) == str:
            log.info(f'Using {weight_init} weight initialization')
            weight_init = get_weight_init_by_string(weight_init)

        if type(bias_init) == str:
            bias_init = get_weight_init_by_string(bias_init)

        if type(activation) is str:
            activation = str2act(activation)

        self.n_atom_basis = self.hidden_dim = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.rbf_cutoff = rbf_cutoff
        self.cg_degree = cg_degree
        if self.cg_degree > 1:
            self.coarse_grain = True
            self.cg_linear = nn.Linear(self.cg_degree*n_atom_basis,n_atom_basis)

        self.neighbor_embedding = NodeInit([self.hidden_dim // 2, self.hidden_dim], n_rbf, self.cutoff, max_z=max_z,cg_degree=self.cg_degree,
                                           weight_init=weight_init, bias_init=bias_init, concat=False,
                                           proj_ln='layer', activation=activation).jittable()
        self.edge_embedding = EdgeInit(n_rbf, [self.hidden_dim // 2, self.hidden_dim], weight_init=weight_init,
                                       bias_init=bias_init,
                                       proj_ln='').jittable()

        radial_basis = str2basis(radial_basis)
        self.radial_basis = radial_basis(cutoff=self.rbf_cutoff, n_rbf=n_rbf)

        self.embedding = nn.Embedding(max_z, n_atom_basis)

        self.tensor_init = TensorInit(l=lmax)

        self.gata = nn.ModuleList([
            GATA(
                n_atom_basis=self.n_atom_basis, activation=activation, aggr=aggr,
                weight_init=weight_init, bias_init=bias_init,
                layer_norm=int_layer_norm, vector_norm=int_vector_norm, cutoff=self.cutoff, epsilon=epsilon,
                num_heads=num_heads, dropout=attn_dropout,
                edge_updates=edge_updates, last_layer=(i == self.n_interactions - 1),
                scale_edge=scale_edge, edge_ln=edge_ln,
                evec_dim=evec_dim, emlp_dim=emlp_dim,
                sep_vecj=sep_int_vec, lmax=lmax
            ).jittable() for i in range(self.n_interactions)
        ])

        self.eqff = nn.ModuleList([
            EQFF(
                n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon,
                weight_init=weight_init, bias_init=bias_init
            ) for i in range(self.n_interactions)
        ])
        self.out_dim = out_dim
        self.rep_linear = nn.Linear(self.hidden_dim, 3)
        self.out_linear = nn.Linear(3,self.out_dim)
        if softmax:
            self.softmax = nn.Softmax(dim=1)
        else:
            self.softmax = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        for l in self.gata:
            l.reset_parameters()
        for l in self.eqff:
            l.reset_parameters()

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (Mapping[str, Tensor]): Dictionary of input tensors containing
            atomic_numbers, pos, batch, edge_index, r_ij, and dir_ij.

        Returns:
            Tuple[Tensor, Tensor]: Returns tuple of atomic representation and intermediate
            atom-wise representation q and mu, of respective shapes
            [num_nodes, 1, hidden_dims] and [num_nodes, 3, hidden_dims].

        """
        # get tensors from input dictionary
        atomic_numbers, batch, edge_index, edge_weight, edge_vec = inputs.z, inputs.batch, inputs.edge_index, inputs.edge_weight, inputs.edge_vec

        q = self.embedding(atomic_numbers)[:]
        if self.coarse_grain:
            assert inputs.cg_index is not None
            idx = inputs.cg_index.long()   # [N]
            K   = int(idx.max()) + 1 # number of cg nodes 
            counts  = torch.bincount(idx, minlength=K)        # [K] cg degree list
            max_len = counts.max().item()
            padded = q.new_zeros((K, max_len, self.n_atom_basis)) # all zero
            row = torch.repeat_interleave(torch.arange(K, device=q.device), counts)
            col = torch.arange(max_len, device=q.device).repeat(K) #[node number]
            mask = col < counts.reshape(1,-1).repeat(max_len,1).transpose(1,0).reshape(-1)             # [K, max_len]
            padded[row, col[mask]] = q                      # scatter into padded
            q = self.cg_linear(padded.reshape(K,-1))
        
        edge_attr = self.radial_basis(edge_weight)
        q = self.neighbor_embedding(atomic_numbers, q, edge_index, edge_weight, edge_attr, inputs.cg_index)
        edge_attr = self.edge_embedding(edge_index, edge_attr, q)
        mask = edge_index[0] != edge_index[1]
        dist = torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec[mask] = edge_vec[mask] / dist

        edge_vec = self.tensor_init(edge_vec)
        equi_dim = ((self.tensor_init.l + 1) ** 2) - 1
        # count number of edges for each node
        num_edges = scatter(torch.ones_like(edge_weight), edge_index[0], dim=0, reduce="sum")
        # the shape of num edges is [num_nodes, 1], we want to expand this to [num_edges, 1]
        # Map num_edges back to the shape of attn using edge_index
        num_edges_expanded = num_edges[edge_index[0]]

        qs = q.shape
        mu = torch.zeros((qs[0], equi_dim, qs[1]), device=q.device)
        q.unsqueeze_(1)
        for i, (interaction, mixing) in enumerate(zip(self.gata, self.eqff)):
            q, mu, edge_attr = interaction(edge_index, q, mu, dir_ij=edge_vec, r_ij=edge_attr, d_ij=edge_weight,
                                           num_edges_expanded=num_edges_expanded)  # idx_i, idx_j, n_atoms, # , f_ij=f_ij
            q, mu = mixing(q, mu)

        q = q.squeeze(1) 
        return q, mu, batch

class out_state_rep(nn.Module):
    def __init__(self, s_in, s_out, n_hidden=None , softmax=True, activation=shifted_softplus):
        super(out_state_rep, self).__init__()
        self.s_in = s_in
        self.v_in = s_in
        self.s_out = s_out
        if n_hidden == None:
            self.n_hidden = s_in
        else:
            self.n_hidden = n_hidden
        self.outnet = nn.ModuleList(
                [
                    GatedEquivariantBlock(n_sin=self.s_in, n_vin=self.v_in, n_sout=self.n_hidden, n_vout=self.n_hidden, n_hidden=self.n_hidden,
                                          activation=activation,
                                          sactivation=activation),
                    GatedEquivariantBlock(n_sin=self.n_hidden, n_vin=self.n_hidden, n_sout=3, n_vout=1,
                                          n_hidden=self.n_hidden, activation=activation)
                ])
        self.out_layer = nn.Linear(3,self.s_out)
        if softmax:
            self.softmax = nn.Softmax(dim=1)
        else:
            self.softmax = nn.Identity()
    def forward(self, input):
        rep_s, rep_v, batch = input
        for layer in self.outnet:
            rep_s, rep_v = layer(rep_s, rep_v)

        out_s = global_mean_pool(rep_s,batch)
        out_s = F.normalize(out_s,dim=-1) # latest version

        return self.softmax(self.out_layer(out_s)), out_s
