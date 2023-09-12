import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim,bias=False)

    def forward(self, adj_matrix, node_features):
        # GCN layer implementation
        print(adj_matrix.shape,node_features.shape)
        adjacency_hat = torch.matmul(adj_matrix, node_features)
        print(adjacency_hat.shape,self.linear.weight.shape)
        output = self.linear(adjacency_hat)
        #output = torch.matmul(adjacency_hat, self.linear.weight)
        return output

class GCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(num_features, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, adj_matrix, node_features):
        h1 = F.relu(self.layer1(adj_matrix, node_features))
        output = self.layer2(adj_matrix, h1)
        return output

if __name__=='__main__':
    # 构造一个简单的图，这里使用邻接矩阵表示
    # 例如，如果有3个节点，邻接矩阵可以表示为：
    # [[0, 1, 1],
    #  [1, 0, 1],
    #  [1, 1, 0]]
    adj_matrix = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float32)
    # 节点特征向量
    node_features = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float32)
    # 创建GCN模型
    model = GCN(num_features=2, num_classes=2, hidden_dim=4)
     # 执行前向传播
    output = model(adj_matrix, node_features)
    print("GCN输出：", output)