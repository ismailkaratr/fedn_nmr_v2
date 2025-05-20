from __future__ import print_function
import sys
import yaml
import torch
import collections
import os
from read_data import read_data
from fedn.utils.helpers.helpers import get_helper, save_metadata
import datetime
# FEDn helper modülünü tanımlıyoruz
HELPER_MODULE = "numpyhelper"

def weights_to_np(weights):
    """Converts PyTorch model weights to numpy arrays for FEDn helper."""
    weights_np = []
    for w in weights:
        weights_np.append(weights[w].cpu().detach().numpy())
    return weights_np

def np_to_weights(weights_np, model):
    """Converts numpy arrays from FEDn helper to PyTorch model weights."""
    state_dict = collections.OrderedDict()
    model_state = model.state_dict()
    
    # Map numpy arrays to PyTorch model parameters
    for i, key in enumerate(model_state.keys()):
        # Make sure we have enough weights
        if i < len(weights_np):
            state_dict[key] = torch.tensor(weights_np[i])
        else:
            raise ValueError(f"Model parameter count mismatch: numpy model has {len(weights_np)} arrays, PyTorch model expects more.")
    
    return state_dict

def train(model, loss, optimizer, data_path, settings):
    print("-- RUNNING TRAINING --", flush=True)
    
    # Read data from path
    print(f"Reading data from: {data_path}", flush=True)
    trainset = read_data(data_path)
    
    print('Starting model training', flush=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=settings['batch_size'], shuffle=True)

    start_time = datetime.datetime.now()

    model.train()
    num_examples_trained = 0
    
    for epoch in range(settings['epochs']):
        epoch_loss = 0
        batch_count = 0
        for x, y in train_loader:
            optimizer.zero_grad()

            batch_size = x.shape[0]
            x = torch.squeeze(x, 1)
            x_float = torch.from_numpy(x.float().numpy())

            output = model.forward(x_float)
            
            input = torch.zeros((batch_size, 128), dtype=torch.float32)
            input_mask = torch.zeros((batch_size, 128), dtype=torch.int32)
            
            # Create mask based on the last column
            for i, row in enumerate(x):
                atom_idx = int(torch.FloatTensor(row)[70401].item()) if row.shape[0] > 70401 else 0
                input_mask[i, atom_idx] = 1
                input[i, atom_idx] = float(y[i].item())

            error = loss(output, input, input_mask)
            error.backward()
            optimizer.step()
            
            epoch_loss += error.item()
            batch_count += 1
            num_examples_trained += batch_size
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{settings['epochs']}, Loss: {epoch_loss/batch_count:.6f}", flush=True)

    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time}", flush=True)

    print("-- TRAINING COMPLETED --", flush=True)
    # Modelimizle ilgili metadata bilgilerini de döndürüyoruz
    return model, num_examples_trained

if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python train.py <input_model> <output_model> [data_path]", flush=True)
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    
    # Determine data path:
    # 1. First check environment variable
    # 2. Use argument if provided
    # 3. Try default paths if neither exists
    data_path = os.environ.get('FEDN_TRAIN_CSV')
    if not data_path and len(sys.argv) > 3:
        data_path = sys.argv[3]
    
    # If no path specified or file doesn't exist, try alternative paths
    if not data_path or not os.path.exists(data_path):
        if data_path:
            print(f"WARNING: Specified file not found: {data_path}", flush=True)
        
        # Try commonly used possible paths
        alt_paths = [
            'output0train.csv',  # User specified format
            '../output0train.csv',
            '../../output0train.csv',
            'train.csv',
            '../train.csv',
            '../data/train.csv',
            '../data/clients/0/train.csv'
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                data_path = path
                print(f"Alternative data path found: {data_path}", flush=True)
                break
        
        if not data_path or not os.path.exists(data_path):
            print(f"WARNING: No data file found. Please specify the correct path to CSV file.", flush=True)
            print(f"Either set the FEDN_TRAIN_CSV environment variable or provide the CSV path as a command line parameter.", flush=True)
            sys.exit(1)
    
    print(f"Input model: {input_model}", flush=True)
    print(f"Output model: {output_model}", flush=True)
    print(f"Data path: {data_path}", flush=True)

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            print(f"Settings file reading error: {e}", flush=True)
            raise(e)

    # FEDn'nin helper sistemini kullanıyoruz
    helper = get_helper(HELPER_MODULE)
    
    from models.pytorch_model import create_seed_model
    
    model, loss, optimizer = create_seed_model(settings)
    print('Model created', flush=True)
    
    # Load model
    try:
        model_weights = helper.load(input_model)
        model.load_state_dict(np_to_weights(model_weights, model))
        print("Model successfully loaded", flush=True)
    except Exception as e:
        print(f"Model loading error: {e}", flush=True)
        print("WARNING: Model could not be loaded, continuing with a new model", flush=True)
    
    # Training process
    model, num_examples_trained = train(model, loss, optimizer, data_path, settings)
    
    # Save model
    try:
        helper.save(weights_to_np(model.state_dict()), output_model)
        print(f"Model saved: {output_model}", flush=True)
        
        # Metadata oluştur ve kaydet
        metadata = {
            "num_examples": num_examples_trained,
            "batch_size": settings['batch_size'],
            "epochs": settings['epochs'],
            "model_type": "pytorch"
        }
        save_metadata(metadata, output_model)
        print(f"Model metadata saved to: {output_model}-metadata", flush=True)
    except Exception as e:
        print(f"Model saving error: {e}", flush=True)
        raise
