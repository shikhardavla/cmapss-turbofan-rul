import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from src.models.base import BaseModel


class PositionalEncoding(nn.Module):
    """
    Implements standard sinusoidal positional encoding for time-series data.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the sinusoidal frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: (1, max_len, d_model) to broadcast across batch sizes
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerNetwork(nn.Module):
    """
    The underlying PyTorch neural network architecture.
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        
        # 1. Map raw features to the hidden dimension size (d_model)
        self.feature_embedding = nn.Linear(input_dim, d_model)
        
        # 2. Add Positional Information
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True # Crucial for accepting (Batch, Seq, Features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Regression Head mapping back to a single RUL prediction
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = self.feature_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Extract the output of the final time step to predict RUL
        # Alternative: x.mean(dim=1) for Global Average Pooling
        final_step = x[:, -1, :] 
        
        out = self.regressor(final_step)

        # Explicitly squeeze only the single output dimension: (Batch, 1) -> (Batch,)
        # Keeps shape consistent with LSTMModel/TCNModel, which both return
        # (Batch,) predictions -- without this, evaluate.py's metric calls
        # (mean_squared_error(y, preds), etc.) would mix a (N,) y_true against
        # an (N, 1) y_pred, risking silent incorrect broadcasting.
        return out.squeeze(-1)


class TransformerModel(BaseModel):
    """
    Wrapper class compatible with BaseModel and sequential.py tensors.
    Follows the same self-contained save/load and CPU-first device pattern
    as LSTMModel/TCNModel.
    """
    def __init__(self, **params):
        self.d_model = params.get('d_model', 64)
        self.nhead = params.get('nhead', 4)
        self.num_layers = params.get('num_layers', 2)
        self.dropout = params.get('dropout', 0.1)
        self.epochs = params.get('epochs', 20)
        self.batch_size = params.get('batch_size', 512)
        self.lr = params.get('lr', 0.001)
        # Fraction of total training steps spent ramping the LR up from
        # near-zero to self.lr, before the existing linear decay kicks in.
        # Transformers are prone to early-training instability (large,
        # poorly-conditioned updates interacting badly with LayerNorm)
        # without this ramp-up.
        self.warmup_ratio = params.get('warmup_ratio', 0.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # Initialized dynamically in train() based on input_dim
        self.input_dim = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Trains the Transformer model using PyTorch.
        X_train shape expected: (Samples, Window, Features)
        """
        # Dynamically define network based on input feature dimension
        self.input_dim = X_train.shape[2]
        self.model = TransformerNetwork(
            input_dim=self.input_dim,
            d_model=self.d_model, 
            nhead=self.nhead, 
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
        
        # Learning Rate Scheduler: short linear warmup (near-zero -> 100% of
        # lr) followed by the existing linear decay (100% -> 10% of lr).
        total_steps = self.epochs * len(loader)
        warmup_steps = max(1, int(self.warmup_ratio * total_steps))
        decay_steps = max(1, total_steps - warmup_steps)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=decay_steps
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                scheduler.step() # Step scheduler per batch
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(loader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts RUL for the given 3D sequences.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        self.model.eval()

        # Keep tensor on CPU; only move each batch to self.device inside the loop
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X[0].to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())

        # Use concatenate, not vstack: outputs are now 1D (Batch,) per
        # batch, and vstack would require every batch (including the final,
        # possibly smaller one) to have equal length.
        return np.concatenate(predictions)

    def save(self, path):
        """Saves weights alongside architecture parameters, so load() is self-contained."""
        state = {
            'model_state': self.model.state_dict(),
            'hyperparams': {
                'input_dim': self.input_dim,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        }
        torch.save(state, path)

    def load(self, path):
        """Loads weights and reconstructs the architecture from saved hyperparameters."""
        state = torch.load(path, map_location=self.device)
        hyperparams = state['hyperparams']

        self.input_dim = hyperparams['input_dim']
        self.d_model = hyperparams['d_model']
        self.nhead = hyperparams['nhead']
        self.num_layers = hyperparams['num_layers']
        self.dropout = hyperparams['dropout']

        self.model = TransformerNetwork(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(state['model_state'])
        return self