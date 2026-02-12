import torch
import torch.nn as nn
import torch.nn.functional as F

class FutureRadarCNN(nn.Module):
    """
    Entrada: (B, 5, H, W)  -> 5 frames como 5 canais
    Saída: probabilidade (B, 1)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # H=W=200 -> 100 -> 50
        self.fc1 = nn.Linear(32 * 50 * 50, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 200 -> 100
        x = self.pool(F.relu(self.conv2(x)))  # 100 -> 50
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
