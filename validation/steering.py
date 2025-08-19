import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import json


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
    
    # Prepare balanced sampling from each task
    task_names = list(tasks.keys())
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        epoch_loss = 0
        num_batches = 0
        
        # Sample balanced mini-batches from each attribute
        min_samples = min(min(tasks[task][0].shape[0], tasks[task][1].shape[0]) for task in tasks)
        effective_batch_size = min(batch_size // (2 * num_attributes), min_samples)
        
        if effective_batch_size < 1:
            effective_batch_size = 1
        
        # Create batches by sampling from each task
        for batch_idx in range(0, min_samples, effective_batch_size):
            batch_activations = []
            batch_labels = []
            batch_task_indices = []
            
            for t, task_name in enumerate(task_names):
                pos_acts, neg_acts = tasks[task_name]
                
                # Sample positive and negative examples
                pos_indices = torch.randperm(pos_acts.shape[0])[:effective_batch_size]
                neg_indices = torch.randperm(neg_acts.shape[0])[:effective_batch_size]
                
                pos_batch = pos_acts[pos_indices]
                neg_batch = neg_acts[neg_indices]
                
                batch_activations.append(pos_batch)
                batch_activations.append(neg_batch)
                
                batch_labels.extend([1] * effective_batch_size)  # positive
                batch_labels.extend([0] * effective_batch_size)  # negative
                
                batch_task_indices.extend([t] * effective_batch_size * 2)
            
            batch_activations = torch.cat(batch_activations, dim=0)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32)
            batch_task_indices = torch.tensor(batch_task_indices, dtype=torch.long)
            
            # Forward pass
            adjusted_acts, gates = model(batch_activations)
            adjusted_acts = normalize_activations(batch_activations, adjusted_acts)
            
            # Compute losses
            total_loss = 0
            
            # MMD loss per attribute
            loss_mmd = 0
            for t, task_name in enumerate(task_names):
                task_mask = batch_task_indices == t
                if task_mask.sum() > 0:
                    task_acts = adjusted_acts[task_mask]
                    task_lbls = batch_labels[task_mask]
                    
                    pos_mask = task_lbls == 1
                    neg_mask = task_lbls == 0
                    
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        pos_adjusted = task_acts[pos_mask]
                        neg_adjusted = task_acts[neg_mask]
                        
                        # Compare adjusted negatives to original positives
                        original_pos = tasks[task_name][0]
                        sample_indices = torch.randperm(original_pos.shape[0])[:min(pos_adjusted.shape[0], original_pos.shape[0])]
                        original_pos_sample = original_pos[sample_indices]
                        
                        loss_mmd += compute_mmd(neg_adjusted, original_pos_sample, sigma)
            
            loss_mmd = loss_mmd / num_attributes
            
            # Sparsity loss on negative examples
            neg_mask = batch_labels == 0
            if neg_mask.sum() > 0:
                loss_sparse = sparsity_loss(gates[neg_mask])
            else:
                loss_sparse = torch.tensor(0.0)
            
            # Preservation loss on positive examples
            pos_mask = batch_labels == 1
            if pos_mask.sum() > 0:
                loss_pos = preservation_loss(gates[pos_mask])
            else:
                loss_pos = torch.tensor(0.0)
            
            # Orthogonality loss
            loss_ortho = orthogonality_loss([sv for sv in model.steering_vectors])
            
            # Combined loss
            batch_loss = (lambda_mmd * loss_mmd + 
                         lambda_sparse * loss_sparse + 
                         lambda_ortho * loss_ortho + 
                         lambda_pos * loss_pos)
            
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        if epoch % 10 == 0:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # Save model with metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_attributes': num_attributes,
        'task_names': task_names
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Specify the model name")
    parser.add_argument("--layer", type=int, default=14, help="Layer index for intervention")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the trained model")
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
    
    if args.save_path is None:
        args.save_path = f"checkpoints/{args.model_name}_L{args.layer}_mat_steer.pt"
    
    # Ensure checkpoints directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    tasks = {}
    for dataset_name in datasets:
        # Load the corrected labels (not token_labels)
        labels = np.load(f'../features/{args.model_name}_{dataset_name}_labels.npy')
        all_layer_wise_activations = np.load(f'../features/{args.model_name}_{dataset_name}_layer_wise.npy')
        
        # Filter by positive/negative labels
        pos_acts = torch.tensor(all_layer_wise_activations[labels == 1], dtype=torch.float32)
        neg_acts = torch.tensor(all_layer_wise_activations[labels == 0], dtype=torch.float32)
        tasks[dataset_name] = (pos_acts, neg_acts)
        
        print(f"Dataset {dataset_name}: {pos_acts.shape[0]} positive, {neg_acts.shape[0]} negative examples")
    
    model = train_multi_task_steering(
        tasks, num_attributes, args.batch_size, args.epochs, args.lr, args.sigma, 
        args.lambda_mmd, args.lambda_sparse, args.lambda_ortho, args.lambda_pos, args.save_path
    )
    
    # Save metadata with the checkpoint
    metadata = {
        'layer': args.layer,
        'model_name': args.model_name,
        'datasets': datasets,
        'hyperparams': {
            'lr': args.lr,
            'sigma': args.sigma,
            'lambda_mmd': args.lambda_mmd,
            'lambda_sparse': args.lambda_sparse,
            'lambda_ortho': args.lambda_ortho,
            'lambda_pos': args.lambda_pos
        }
    }
    
    metadata_path = args.save_path.replace('.pt', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Training completed for all datasets.")


if __name__ == "__main__":
    main()