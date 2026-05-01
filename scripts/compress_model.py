import torch
import torch.nn.utils.prune as prune
import os
import joblib
from models.definitions import StudentModel

def compress_model():
    input_dim = np.load('data/processed/X_train.npy').shape[1]
    target_le = joblib.load('data/processed/target_le.joblib')
    num_classes = len(target_le.classes_)
    
    device = torch.device('cpu') # Quantization often works on CPU
    model = StudentModel(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load('models/saved/student_model.pth', map_location=device))
    
    # 1. Measure Initial Size
    torch.save(model.state_dict(), 'models/saved/temp_orig.pth')
    orig_size = os.path.getsize('models/saved/temp_orig.pth') / 1024
    
    # 2. Pruning
    # Prune 30% of weights in FC layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, 'weight') # Make pruning permanent
            
    # 3. Dynamic Quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # 4. Save & Measure
    torch.save(quantized_model.state_dict(), 'models/saved/student_compressed.pth')
    comp_size = os.path.getsize('models/saved/student_compressed.pth') / 1024
    
    print(f"Original Model Size: {orig_size:.2f} KB")
    print(f"Compressed Model Size: {comp_size:.2f} KB")
    print(f"Reduction: {((orig_size - comp_size) / orig_size) * 100:.2f}%")
    
    # Clean up
    if os.path.exists('models/saved/temp_orig.pth'):
        os.remove('models/saved/temp_orig.pth')

import numpy as np
if __name__ == "__main__":
    compress_model()
