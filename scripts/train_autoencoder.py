import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from models.definitions import Autoencoder

def train_autoencoder():
    # 1. Load Normal Data Only
    X_train_normal = np.load('data/processed/X_train_normal.npy').astype(np.float32)
    input_dim = X_train_normal.shape[1]
    
    dataset = TensorDataset(torch.from_numpy(X_train_normal))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 2. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 30
    print(f"Starting Autoencoder training (Normal samples: {len(X_train_normal)}) on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, in loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, X_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Reconstruction Loss: {total_loss/len(loader):.6f}")
            
    # 4. Save
    torch.save(model.state_dict(), 'models/saved/autoencoder.pth')
    
    # 5. Determine Threshold (e.g., 95th percentile of normal reconstruction error)
    model.eval()
    errors = []
    with torch.no_grad():
        for X_batch, in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            loss = torch.mean((outputs - X_batch)**2, dim=1)
            errors.extend(loss.cpu().numpy())
            
    threshold = np.percentile(errors, 90) # Lower to be more sensitive
    np.save('models/saved/anomaly_threshold.npy', threshold)
    print(f"Autoencoder saved. Threshold set to: {threshold:.6f}")

if __name__ == "__main__":
    train_autoencoder()
