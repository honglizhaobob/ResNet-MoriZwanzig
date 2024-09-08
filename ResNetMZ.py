import numpy as np
import torch

class ResNet(torch.nn.Module):
    def __init__(self, n_mem, model=None):
        super().__init__()
        self.input_size = n_mem + 1 # z_t, z_t-1, ..., zt-n_mem
        if model is not None:
            self.model = model
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, 30),
                torch.nn.ReLU(),
                torch.nn.Linear(30, 30),
                torch.nn.ReLU(),
                torch.nn.Linear(30, 30),
                torch.nn.ReLU(),
                torch.nn.Linear(30, 30),
                torch.nn.ReLU(),
                torch.nn.Linear(30, 1)
            )

    def forward(self, x):
        assert x.shape[1] == self.input_size
        x = x.float()
        # last
        x_curr = x[:, -1].reshape(-1, 1).float()
        memory_integral = self.model(x)
        return x_curr + memory_integral

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def predict(self, x_init, num_predictions, print_freq=100):
        """ Given an initial memory, generate forward predictions. """
        N, M = x_init.shape
        assert M == self.input_size, "Incorrect input dimensions. "
        # initial memory
        memory = torch.tensor(x_init, dtype=torch.float64)
        # current predicted path
        path = memory
        for i in np.arange(1, num_predictions+1):
            if print_freq is not None:
                if i % print_freq == 0:
                    print("Predicting forward {} steps ...".format(i))
            # make prediction and append
            path = torch.cat([path, self.forward(memory)], dim=1)
            # shift memory
            memory = path[:, i:i+self.input_size]
        return path


## Torch dataset
# data set
class ReducedOrderDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, requires_grad=True)
        self.y = torch.tensor(y, requires_grad=True)

    def __len__(self):
        assert self.X.shape[0] == len(self.y)
        return len(self.y)
    
    def __getitem__(self, idx):
        sample = self.X[idx, :]
        sample_response = self.y[idx].reshape(-1, 1)
        return (sample, sample_response)