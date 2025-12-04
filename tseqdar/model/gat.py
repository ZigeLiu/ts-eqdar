import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils._trim_to_layer import TrimToLayer
from torch_scatter import scatter
import torch.nn.functional as F


class GAT(BasicGNN):
    """The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable] = 'relu',
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[Union[str, Callable]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            jk=jk,
            **kwargs,
        )
        self.node_embedding = torch.nn.Embedding(100,in_channels)
        self.pool_nodes = kwargs.get('pool_nodes', True)
        self.lin1 = torch.nn.Linear(hidden_channels, 3)
        self.lin2 = torch.nn.Linear(3, out_channels) 
        self.softmax = torch.nn.Softmax(dim=-1)


    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs)

    def forward(
        self,
        data: Data,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        x, edge_index, edge_attr = data.z, data.edge_index, data.edge_weight
        edge_weight = None
        self.supports_edge_weight = False
        self.supports_edge_attr = True if edge_attr is not None else False
        batch = data.batch if data.batch is not None else torch.zeros(len(x), dtype=torch.long)
        batch_size = int(batch.max().item()) + 1 if batch is not None else 1
        x = self.node_embedding(x.long())

        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if (not torch.jit.is_scripting()
                    and num_sampled_nodes_per_hop is not None):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, 'jk'):
                    xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x # [nodes, hidden channels]
        x = self.lin1(x) # [nodes, 3]

        if self.pool_nodes:
            x = scatter(x, batch, dim=0, reduce="sum")
            x = F.normalize(x,dim=-1)
            return self.softmax(self.lin2(x)), x
        else:
            return x



