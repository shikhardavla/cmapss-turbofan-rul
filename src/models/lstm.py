import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.base import BaseModel


class PyTorchLSTM(nn.Module):
    """
    The internal PyTorch neural network architecture.
    """
    def __init__(self, input_dim, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        
        # batch_first=True means inputs are [Batch, Time_Steps, Features]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Decoder head
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1) # Single RUL output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Extract last time step state
        last_step_out = lstm_out[:, -1, :] 
        
        out = self.fc1(last_step_out)
        out = self.relu(out)
        out = self.fc2(out)
        
        # Explicitly squeeze only the single output dimension: (Batch, 1) -> (Batch,)
        return out.squeeze(-1)


class LSTMModel(BaseModel):
    """
    Production-ready wrapper for LSTM sequence modeling.
    """
    def __init__(self, **params):
        self.hidden_size = params.get('hidden_size', 64)
        self.num_layers = params.get('num_layers', 2)
        self.dropout = params.get('dropout', 0.2)
        self.lr = params.get('lr', 0.001)
        self.epochs = params.get('epochs', 50)
        self.batch_size = params.get('batch_size', 256)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_names = None

    def train(self, X_train, y_train, **kwargs):
        input_dim = X_train.shape[2] 
        
        self.model = PyTorchLSTM(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Keep tensors on CPU; only move each batch to self.device inside
        # the loop below. This avoids allocating the full dataset on GPU
        # memory at once, and keeps behavior consistent regardless of
        # whether CUDA is available.
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_X)
                
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                avg_loss = epoch_loss / len(dataset)
                print(f"LSTM Epoch {epoch:02d} | Loss (MSE): {avg_loss:.4f}")

    def predict(self, X):
        """Batched predictions to prevent CUDA OOM on large test datasets."""
        self.model.eval()
        
        # Keep tensor on CPU; only move each batch to self.device inside the loop
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(self.device)
                predictions = self.model(batch_X)
                all_preds.append(predictions.cpu().numpy())
            
        return np.concatenate(all_preds)

    def save(self, path):
        """Saves weights and architecture parameters."""
        state = {
            'model_state': self.model.state_dict(),
            'hyperparams': {
                'input_dim': self.model.lstm.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            },
            'feature_names': self.feature_names
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, device=None):
        """Factory method to load saved model from disk."""
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=target_device)
        
        params = state['hyperparams']
        instance = cls(
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params.get('dropout', 0.2)
        )
        
        instance.device = target_device
        instance.model = PyTorchLSTM(
            input_dim=params['input_dim'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params.get('dropout', 0.2)
        ).to(instance.device)
        
        instance.model.load_state_dict(state['model_state'])
        instance.feature_names = state.get('feature_names', None)
        return instance