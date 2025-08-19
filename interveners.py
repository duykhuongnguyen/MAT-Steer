import torch
import torch.nn as nn
import torch.nn.functional as F

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False  
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        if self.head == -1:
            self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        return b
    
class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        self.states.append(b[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b


class MATIntervener():
    """Multi-Attribute Steering Intervener for runtime intervention."""
    collect_state = True
    collect_action = True
    
    def __init__(self, steering_vectors, gates_weights, gates_biases=None, multiplier=1.0, layer_norm_preserve=True):
        """
        Args:
            steering_vectors: tensor of shape (num_attributes, hidden_dim) 
            gates_weights: tensor of shape (num_attributes, hidden_dim)
            gates_biases: optional tensor of shape (num_attributes,)
            multiplier: global scaling factor for intervention strength
            layer_norm_preserve: whether to preserve the original L2 norm
        """
        self.steering_vectors = steering_vectors  # (T, D)
        self.gates_weights = gates_weights  # (T, D) 
        self.gates_biases = gates_biases if gates_biases is not None else torch.zeros(steering_vectors.shape[0])  # (T,)
        self.multiplier = multiplier
        self.layer_norm_preserve = layer_norm_preserve
        self.states = []
        self.actions = []
        
    def reset(self):
        self.states = []
        self.actions = []
        
    def __call__(self, b, s):
        """Apply MAT-Steer intervention.
        
        Args:
            b: input tensor of shape (batch_size, seq_len, hidden_dim)
            s: additional state (unused)
        """
        # Extract the last token activation
        x = b[0, -1]  # (hidden_dim,)
        self.states.append(x.detach().clone())
        
        # Move tensors to the same device as input
        device = x.device
        steering_vectors = self.steering_vectors.to(device)
        gates_weights = self.gates_weights.to(device)
        gates_biases = self.gates_biases.to(device)
        
        # Compute gates: g_t = sigmoid(W_t @ x + b_t) for each attribute t
        gates = torch.sigmoid(torch.matmul(gates_weights, x) + gates_biases)  # (num_attributes,)
        
        # Compute steering delta: sum_t g_t * v_t
        delta = torch.sum(gates.unsqueeze(1) * steering_vectors, dim=0)  # (hidden_dim,)
        
        # Apply intervention with global multiplier
        intervention = self.multiplier * delta
        self.actions.append(intervention.detach().clone())
        
        # Apply intervention
        x_adjusted = x + intervention
        
        # Preserve original norm if requested
        if self.layer_norm_preserve:
            original_norm = torch.norm(x, p=2)
            adjusted_norm = torch.norm(x_adjusted, p=2)
            if adjusted_norm > 1e-8:  # Avoid division by zero
                x_adjusted = x_adjusted * (original_norm / adjusted_norm)
        
        # Update the tensor in-place
        b[0, -1] = x_adjusted
        
        return b

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, multiplier=1.0, layer_norm_preserve=True):
        """Load MATIntervener from a saved checkpoint.
        
        Args:
            checkpoint_path: path to the saved model checkpoint
            multiplier: global scaling factor
            layer_norm_preserve: whether to preserve L2 norm
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            num_attributes = checkpoint['num_attributes']
            input_dim = checkpoint['input_dim']
        else:
            # Legacy format
            state_dict = checkpoint
            # Try to infer dimensions from state_dict
            steering_vectors_keys = [k for k in state_dict.keys() if 'steering_vectors' in k]
            num_attributes = len(steering_vectors_keys)
            input_dim = state_dict[steering_vectors_keys[0]].shape[0]
        
        # Extract steering vectors
        steering_vectors = torch.stack([
            state_dict[f'steering_vectors.{i}'] for i in range(num_attributes)
        ])  # (num_attributes, input_dim)
        
        # Extract gating weights
        gates_weights = torch.stack([
            state_dict[f'gating_weights.{i}.weight'].squeeze() for i in range(num_attributes)
        ])  # (num_attributes, input_dim)
        
        # Extract gating biases (if available)
        gates_biases = torch.stack([
            state_dict[f'gating_weights.{i}.bias'] for i in range(num_attributes)
        ])  # (num_attributes,)
        
        return cls(steering_vectors, gates_weights, gates_biases, multiplier, layer_norm_preserve)


def create_mat_pyvene_config(layers, mat_intervener):
    """Create pyvene config for MAT intervention at specified layers.
    
    Args:
        layers: list of layer indices to intervene at
        mat_intervener: MATIntervener instance
        
    Returns:
        list of pyvene config dictionaries
    """
    pv_config = []
    for layer in layers:
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(mat_intervener),
        })
    return pv_config