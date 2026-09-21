import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.base import BaseModel


class Chomp1d(nn.Module):
    """
    This class removes the extra padding on the right side of the output to ensure 
    convolutions are strictly causal, i.e., because kernels are symmetric, the
    that they do not look at future instances when training (this is effectively data leakage)
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class PyTorchTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(PyTorchTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, 
                dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            ))

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # PyTorch Conv1d expects shape: (Batch, Channels, Length)
        # We pass in (Batch, Length, Sensors), so we must permute
        x = x.permute(0, 2, 1) 
        
        y1 = self.network(x)
        
        # Take the output of the last time step for RUL prediction
        last_step_output = y1[:, :, -1] 

        # Explicitly squeeze only the single output dimension: (Batch, 1) -> (Batch,)
        # Using squeeze() with no argument would also collapse the batch
        # dimension itself whenever batch size == 1 (e.g. a leftover final
        # batch), producing a 0-d tensor instead of shape (1,).
        return self.linear(last_step_output).squeeze(-1)

# ---------------------------------------------------------
# TCN Wrapper
# ---------------------------------------------------------

class TCNModel(BaseModel):
    def __init__(self, **params):
        self.num_channels = params.get('num_channels', [16, 32, 64])
        self.kernel_size = params.get('kernel_size', 3)
        self.epochs = params.get('epochs', 20)
        self.batch_size = params.get('batch_size', 1024)
        self.lr = params.get('lr', 0.001)
        
        # Safely assign device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_dim = None

    def train(self, X_train, y_train, **kwargs):
        # X_train shape: (samples, window_size, sensors)
        self.input_dim = X_train.shape[2]
        
        self.model = PyTorchTCN(
            num_inputs=self.input_dim, 
            num_channels=self.num_channels, 
            kernel_size=self.kernel_size
        ).to(self.device)

        # Keep data on CPU initially to prevent Out-Of-Memory (OOM) errors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                # Move batches to GPU during the loop
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"TCN Epoch {epoch} | Loss: {epoch_loss/len(loader):.4f}")

    def predict(self, X):
        self.model.eval()
        # Keep tensor on CPU
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        # Do not shuffle for predictions!
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch_X in loader:
                # Extract the tensor from the tuple returned by TensorDataset
                batch_X = batch_X[0].to(self.device)
                predictions = self.model(batch_X)
                all_preds.append(predictions.cpu().numpy())
        
        # Concatenate all batched predictions
        return np.concatenate(all_preds)

    def save(self, path):
        """Saves weights alongside architecture parameters, so load() is self-contained."""
        state = {
            'model_state': self.model.state_dict(),
            'hyperparams': {
                'input_dim': self.input_dim,
                'num_channels': self.num_channels,
                'kernel_size': self.kernel_size,
            }
        }
        torch.save(state, path)

    def load(self, path):
        """Loads weights and reconstructs the architecture from saved hyperparameters."""
        state = torch.load(path, map_location=self.device)
        hyperparams = state['hyperparams']

        self.input_dim = hyperparams['input_dim']
        self.num_channels = hyperparams['num_channels']
        self.kernel_size = hyperparams['kernel_size']

        self.model = PyTorchTCN(
            num_inputs=self.input_dim, 
            num_channels=self.num_channels, 
            kernel_size=self.kernel_size
        ).to(self.device)
        self.model.load_state_dict(state['model_state'])
        return self