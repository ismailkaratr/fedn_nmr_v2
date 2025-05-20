import yaml
import sys
import os
import torch
import collections
import numpy as np
from fedn.utils.helpers.helpers import get_helper, save_metadata

# FEDn helper modülünü tanımlıyoruz
HELPER_MODULE = "numpyhelper"

def weights_to_np(weights):
    """Converts PyTorch model weights to numpy arrays"""
    weights_np = []
    for w in weights:
        weights_np.append(weights[w].cpu().detach().numpy())
    return weights_np

if __name__ == '__main__':
    # Get output file path
    output_model = sys.argv[1] if len(sys.argv) > 1 else 'seed.npz'
    print(f"Creating seed model: {output_model}", flush=True)
    
    # Read settings
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
            print("Settings loaded.", flush=True)
        except yaml.YAMLError as e:
            print(f"Settings file reading error: {e}", flush=True)
            raise(e)
    
    # Create model
    try:
        # FEDn'nin helper sistemini kullanıyoruz
        helper = get_helper(HELPER_MODULE)
        
        from models.pytorch_model import create_seed_model
        
        model, loss, optimizer = create_seed_model(settings)
        print("Seed model successfully created.", flush=True)
        
        # Model ağırlıklarını numpy listesine dönüştürüp helper ile kaydediyoruz
        weights_np = weights_to_np(model.state_dict())
        helper.save(weights_np, path=output_model)
        print(f"Seed model saved: {output_model}", flush=True)
        
        # Seed model için metadata oluştur ve kaydet
        metadata = {
            "num_examples": 0,  # Seed model için örnek sayısı 0
            "batch_size": settings.get('batch_size', 32),
            "epochs": 0,  # Seed model eğitilmediği için epoch sayısı 0
            "model_type": "pytorch"
        }
        save_metadata(metadata, output_model)
        print(f"Seed model metadata saved to: {output_model}-metadata", flush=True)
    except Exception as e:
        print(f"Seed model creation error: {e}", flush=True)
        raise