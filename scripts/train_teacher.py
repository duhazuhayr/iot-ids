import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import joblib
from models.definitions import TeacherModel
from sklearn.metrics import classification_report

def train_teacher():
    # 1. Load Data
    X_train = np.load('data/processed/X_train.npy').astype(np.float32)
    y_train = np.load('data/processed/y_train.npy').astype(np.longlong)
    X_test = np.load('data/processed/X_test.npy').astype(np.float32)
    y_test = np.load('data/processed/y_test.npy').astype(np.longlong)
    
    target_le = joblib.load('data/processed/target_le.joblib')
    num_classes = len(target_le.classes_)
    input_dim = X_train.shape[1]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 2. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TeacherModel(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 20
    print(f"Starting Teacher training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    # 4. Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    print("Teacher Classification Report:")
    print(classification_report(y_test, all_preds, target_names=target_le.classes_))
    
    # 5. Save
    if not os.path.exists('models/saved'):
        os.makedirs('models/saved')
    torch.save(model.state_of_dict() if hasattr(model, 'state_of_dict') else model.state_dict(), 'models/saved/teacher_model.pth')
    print("Teacher model saved to models/saved/teacher_model.pth")

if __name__ == "__main__":
    train_teacher()
