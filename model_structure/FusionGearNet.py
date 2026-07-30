import os
import torch
import sys
from torch import nn
from torchdrug import core, layers
from torchdrug.models.gearnet import GeometryAwareRelationalGraphNeuralNetwork as GearNet

class EnzymeFusionNetwork(nn.Module, core.Configurable):
    def __init__(self, input_dim, hidden_dims=[512, 512, 512], use_graph_construction_model=True, structure_model='gearnet'):
        super(EnzymeFusionNetwork, self).__init__()
        self.structure_model = GearNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            batch_norm=True,
            short_cut=True,
            concat_hidden=True,
            readout="sum",
            num_relation=7,
        )

        self.graph_construction_model = (
            layers.geometry.graph.GraphConstruction(
                node_layers=[layers.geometry.AlphaCarbonNode()],    # graph的atom只保留alpha carbon
                edge_layers=[
                    layers.geometry.SequentialEdge(max_distance=2),     # 序列相邻的残基构建边
                    layers.geometry.SpatialEdge(radius=10, min_distance=5),     # 根据欧氏距离，为邻域结点构建边
                    layers.geometry.KNNEdge(k=10, max_distance=0),      # 根据knn构建边
                ],
            )
            if use_graph_construction_model
            else None
        )

    def forward(self, graph, residue_embedding, enzyme_embedding, all_loss=None, metric=None):

        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        output2 = self.structure_model(graph, residue_embedding, all_loss, metric)
        node_output2 = output2.get("node_feature", output2.get("residue_feature"))

        node_feature = torch.cat([residue_embedding, node_output2], dim=-1)
        graph_feature = torch.cat(
            [enzyme_embedding, output2["graph_feature"]], dim=-1
        )
        return {"graph_feature": graph_feature, "node_feature": node_feature}
