# =================================================
# 模型定义模块
# =================================================
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch.nn.functional as F
import torch.nn as nn
class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

class GATModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=2, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 2, num_classes, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGEModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv, APPNP
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# =================================================
# 增强版模型定义
# =================================================
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

class EnhancedGATModel(nn.Module):
    def __init__(self, num_features, hidden_dims, num_classes, dropout=0.5):
        super(EnhancedGATModel, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dims[0], heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dims[0] * 4, hidden_dims[1], heads=2, dropout=dropout)
        self.conv3 = GATConv(hidden_dims[1] * 2, num_classes, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class SGCModel(nn.Module):
    """Simple Graph Convolutional Network"""
    def __init__(self, num_features, hidden_dim, num_classes, K=2):
        super(SGCModel, self).__init__()
        self.conv1 = SGConv(num_features, hidden_dim, K=K)
        self.conv2 = SGConv(hidden_dim, num_classes, K=K)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

class APPNPModel(nn.Module):
    """Approximate Personalized Propagation of Neural Predictions"""
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(APPNPModel, self).__init__()
        self.lin1 = nn.Linear(num_features, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.prop1 = APPNP(K=10, alpha=0.1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)