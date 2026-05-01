import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import joblib
from models.definitions import TeacherModel, StudentModel

def distillation_loss(student_logits, teacher_logits, labels, T=2, alpha=0.5):
    # Standard Cross Entropy Loss
    hard_loss = nn.functional.cross_entropy(student_logits, labels)
    
    # Soft Loss (Distillation)
    # T (Temperature) softens the probability distributions
    soft_loss = nn.functional.kl_div(
        torch.log_softmax(student_logits / T, dim=1),
        torch.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    return alpha * hard_loss + (1.0 - alpha) * soft_loss

def train_student_distillation():
    # 1. Load Data
    X_train = np.load('data/processed/X_train.npy').astype(np.float32)
    y_train = np.load('data/processed/y_train.npy').astype(np.longlong)
    target_le = joblib.load('data/processed/target_le.joblib')
    
    input_dim = X_train.shape[1]
    num_classes = len(target_le.classes_)
    
    # 2. Load Models
    device = torch.device('cpu') # Use CPU for compatibility, CUDA if available
    
    teacher = TeacherModel(input_dim, num_classes).to(device)
    teacher.load_state_dict(torch.load('models/saved/teacher_model.pth', map_location=device))
    teacher.eval() # Teacher is always in eval mode
    
    student = StudentModel(input_dim, num_classes).to(device)
    
    # 3. Setup Training
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # Increasing batch size for stability
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    # INCREASED EPOCHS for higher precision
    epochs = 20 # Up from 10
    
    print(f"Starting Knowledge Distillation (Student Training for {epochs} epochs)...")
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Predict
            with torch.no_grad():
                teacher_logits = teacher(batch_X)
            
            student_logits = student(batch_X)
            
            # Calculate Distillation Loss
            loss = distillation_loss(student_logits, teacher_logits, batch_y, T=3.0, alpha=0.3)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    # 4. Save Student
    if not os.path.exists('models/saved'):
        os.makedirs('models/saved')
    torch.save(student.state_dict(), 'models/saved/student_model.pth')
    print("Student model (Distilled) saved successfully.")

if __name__ == "__main__":
    train_student_distillation()
