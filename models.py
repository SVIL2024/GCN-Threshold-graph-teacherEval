# =================================================
# 模型定义模块
# =================================================
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch.nn.functional as F
import torch.nn as nn
class EnhancedGCNModel(nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes, dropout=0.5):
        super(EnhancedGCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], num_classes)
        self.dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[0])
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims[1])

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)
