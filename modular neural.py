import torch
import torch.nn as nn

class Module1(nn.Module):
    def __init__(self, input_size, output_size):
        super(Module1, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Module2(nn.Module):
    def __init__(self, input_size, output_size):
        super(Module2, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ModularNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ModularNN, self).__init__()
        self.module1 = Module1(input_size, output_size)
        self.module2 = Module2(input_size, output_size)
    
    def forward(self, x):
        out1 = self.module1(x)
        out2 = self.module2(x)
        return out1, out2
