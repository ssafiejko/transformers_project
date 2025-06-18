import torch
from torch import nn

def basic_mlp(num_classes):
    device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")

    model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        ).to(device)
    
    return model