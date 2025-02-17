import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse


def gaussian_kernel(x, y, sigma):
    pairwise_dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-pairwise_dist / (2 * sigma ** 2))


def compute_mmd(x, y, sigma):
    K_xx = gaussian_kernel(x, x, sigma).mean()
    K_yy = gaussian_kernel(y, y, sigma).mean()
    K_xy = gaussian_kernel(x, y, sigma).mean()
    return K_xx + K_yy - 2 * K_xy


class SteeringModule(nn.Module):
    def __init__(self, input_dim, num_attributes):
        super(SteeringModule, self).__init__()
        self.num_attributes = num_attributes
        
        # Steering vectors 
        self.steering_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim)) for _ in range(num_attributes)
        ])
        
        # Gating function
        self.gating_weights = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_attributes)
        ])
    
    def forward(self, activations):
        adjusted_activations = activations.clone()
        gates = []
        for t in range(self.num_attributes):
            gate = torch.sigmoid(self.gating_weights[t](activations))
            gates.append(gate)
            adjusted_activations += gate * self.steering_vectors[t]
        return adjusted_activations, torch.cat(gates, dim=1)


def normalize_activations(original, adjusted):
    """Normalize the adjusted activations to preserve the original norm."""
    norm_original = torch.norm(original, p=2, dim=1, keepdim=True)
    norm_adjusted = torch.norm(adjusted, p=2, dim=1, keepdim=True) + 1e-8  # Avoid division by zero
    return adjusted * (norm_original / norm_adjusted)


def sparsity_loss(gates):
    """Enforce sparsity in gating activations."""
    return torch.mean(torch.abs(gates))


def orthogonality_loss(steering_vectors):
    """Steering vectors to be orthogonal to minimize interference."""
    loss = 0
    num_vectors = len(steering_vectors)
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            loss += (torch.dot(steering_vectors[i], steering_vectors[j]) / 
                     (torch.norm(steering_vectors[i]) * torch.norm(steering_vectors[j]))) ** 2
    return loss


def preservation_loss(gates):
    """Minimal intervention for positive activations."""
    return torch.mean((gates ** 2))


def train_multi_task_steering(tasks, num_attributes, batch_size, epochs, lr, sigma, lambda_mmd, lambda_sparse, lambda_ortho, lambda_pos, save_path):
    input_dim = list(tasks.values())[0][0].shape[1]  
    model = SteeringModule(input_dim, num_attributes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Constructing balanced mini-batches
    all_pos = torch.cat([tasks[task][0] for task in tasks], dim=0)
    all_neg = torch.cat([tasks[task][1] for task in tasks], dim=0)
    
    dataset = torch.cat([all_pos, all_neg], dim=0)
    labels = torch.cat([torch.ones(all_pos.shape[0]), torch.zeros(all_neg.shape[0])], dim=0)
    indices = torch.randperm(dataset.shape[0])
    dataset, labels = dataset[indices], labels[indices]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss = 0
        for i in range(0, dataset.shape[0], batch_size):
            batch_acts = dataset[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            adjusted_acts, gates = model(batch_acts)
            adjusted_acts = normalize_activations(batch_acts, adjusted_acts)  # Normalize activations
            
            pos_acts = adjusted_acts[batch_labels == 1]
            neg_acts = adjusted_acts[batch_labels == 0]
            
            # Compute MMD loss 
            loss_mmd = sum(compute_mmd(tasks[task][0], normalize_activations(tasks[task][1], model(tasks[task][1])[0]), sigma) for task in tasks) / len(tasks)
            loss_sparse = sparsity_loss(gates[batch_labels == 0])
            loss_ortho = orthogonality_loss([sv for sv in model.steering_vectors])
            loss_pos = preservation_loss(gates[batch_labels == 1])
            
            batch_loss = lambda_mmd * loss_mmd + lambda_sparse * loss_sparse + lambda_ortho * loss_ortho + lambda_pos * loss_pos
            batch_loss.backward()
            total_loss += batch_loss.item()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Specify the model name")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=2.0, help="Sigma value for Gaussian kernel in MMD loss")
    parser.add_argument("--lambda_mmd", type=float, default=1.0, help="Weight for MMD loss")
    parser.add_argument("--lambda_sparse", type=float, default=0.9, help="Weight for sparsity loss")
    parser.add_argument("--lambda_ortho", type=float, default=0.1, help="Weight for orthogonality loss")
    parser.add_argument("--lambda_pos", type=float, default=0.9, help="Weight for preservation loss")
    args = parser.parse_args()
    
    datasets = ["truthfulqa", "toxigen", "bbq"]
    num_attributes = len(datasets)
    
    tasks = {}
    for dataset_name in datasets:
        token_labels = np.load(f'../features/{args.model_name}_{dataset_name}_token_labels.npy')
        all_layer_wise_activations = np.load(f'../features/{args.model_name}_{dataset_name}_layer_wise.npy')
        pos_acts = torch.tensor(all_layer_wise_activations[token_labels == 1], dtype=torch.float32)
        neg_acts = torch.tensor(all_layer_wise_activations[token_labels == 0], dtype=torch.float32)
        tasks[dataset_name] = (pos_acts, neg_acts)
    
    train_multi_task_steering(tasks, num_attributes, args.batch_size, args.epochs, args.lr, args.sigma, args.lambda_mmd, args.lambda_sparse, args.lambda_ortho, args.lambda_pos, save_path=f"{args.model_name}_multi_qa.pth")
    print("Training completed for all datasets.")


if __name__ == "__main__":
    main()