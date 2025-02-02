from torch import nn
import torch.nn.functional as F 

class MetaModel(nn.Module):
    def __init__(self, hidden_dim):
        super(MetaModel, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.hidden(x)

class TaskModel(nn.Module):
    def __init__(self, meta_model, input_dim, output_dim):
        super(TaskModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, meta_model.hidden[0].in_features)
        self.meta_model = meta_model
        self.output_layer = nn.Linear(meta_model.hidden[-2].out_features, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.meta_model(x)
        x = self.output_layer(x)
        return x
    


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class Shared(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super(Shared, self).__init__()
        self.shared = nn.Sequential(         
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.shared(x)
    
