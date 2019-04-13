import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,hidden_layers,seed,drop_p=0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size,hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1,h2 in zip(hidden_layers[:-1], hidden_layers[1:])]) 
        
        self.output = nn.Linear(hidden_layers[-1],action_size)
        #self.dropout = nn.Dropout(p=drop_p)
                                             
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            #x = self.dropout(x)
        
        x = self.output(x)
        return x