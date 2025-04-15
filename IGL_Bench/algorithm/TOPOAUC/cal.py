import torch

def compute_ppr_and_gpr(dataset, pr_prob):
    def index2dense(edge_index, num_nodes):
        A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        A[edge_index[0], edge_index[1]] = 1
        return A

    edge_index = dataset.edge_index
    num_nodes = dataset.num_nodes
    labels = dataset.y
    train_index = torch.tensor(dataset.train_index, device=edge_index.device, dtype=torch.long)

    num_classes = labels.max().item() + 1
    train_nodes_per_class = []
    for cls in range(num_classes):
        class_train_nodes = train_index[labels[train_index] == cls].tolist()
        train_nodes_per_class.append(class_train_nodes)

    A = index2dense(edge_index, num_nodes)
    A_hat = A + torch.eye(num_nodes, device=edge_index.device)  
    D = torch.diag(torch.sum(A_hat, dim=1))  
    D = D.inverse().sqrt()  
    A_hat = torch.mm(torch.mm(D, A_hat), D)  

    I = torch.eye(num_nodes, device=edge_index.device)
    PPR = pr_prob * ((I - (1 - pr_prob) * A_hat).inverse())

    gpr_matrix = []
    for class_nodes in train_nodes_per_class:
        class_nodes_tensor = torch.tensor(class_nodes, device=edge_index.device, dtype=torch.long)
        class_ppr = PPR[class_nodes_tensor] 
        class_gpr = torch.mean(class_ppr, dim=0).squeeze()  
        gpr_matrix.append(class_gpr)

    GPR = torch.stack(gpr_matrix, dim=0).transpose(0, 1) 

    return PPR.cpu(), GPR.cpu()