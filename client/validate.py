import sys
import os
from read_data import read_data
import json
import yaml
import torch
import collections
from fedn.utils.helpers.helpers import get_helper, save_metrics
import numpy as np

# Define FEDn helper module
HELPER_MODULE = "numpyhelper"

def np_to_weights(weights_np, model):
    """Convert weights to model compatible format"""
    state_dict = collections.OrderedDict()
    
    # If weights_np is already a dictionary
    if isinstance(weights_np, dict):
        for key in weights_np:
            state_dict[key] = torch.tensor(weights_np[key])
        return state_dict
    
    # If weights_np is a list, match using current model state_dict
    try:
        model_state = model.state_dict()
        for i, key in enumerate(model_state.keys()):
            if i < len(weights_np):
                state_dict[key] = torch.tensor(weights_np[i])
    except Exception as e:
        print(f"Error converting weights: {e}", flush=True)
        # If failed, return original structure
        return weights_np
    
    return state_dict

def validate(model, data_path, loss, settings):
    print("-- RUNNING VALIDATION --", flush=True)
    print(f"Reading validation data from: {data_path}", flush=True)

    def evaluate(model, loss, dataloader):
        model.eval()
        meanSquaredError = 0
        meanAverageError = 0
        total_samples = 0
        processed_batches = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                print("Processing batch...", flush=True)
                batch_size = x.shape[0]
                total_samples += batch_size
                x = torch.squeeze(x, 1)
                x_float = torch.from_numpy(x.float().numpy())

                print("Running model forward pass...", flush=True)
                # Check model type and output type for debugging
                print(f"Model type: {type(model)}", flush=True)
                output = model.forward(x_float)
                print(f"Output type: {type(output)}", flush=True)
                if isinstance(output, dict):
                    print("Output is a dictionary, extracting tensor...", flush=True)
                    # If output is a dict, usually accessed with keys like 'out' or 'logits'
                    if 'out' in output:
                        output = output['out']
                    elif 'logits' in output:
                        output = output['logits']
                    elif 'predictions' in output:
                        output = output['predictions']
                    else:
                        # Get first tensor from dictionary
                        for key, value in output.items():
                            if isinstance(value, torch.Tensor):
                                output = value
                                print(f"Using key: {key}", flush=True)
                                break
                    
                print("Preparing input tensors...", flush=True)
                input = torch.zeros((batch_size, 128), dtype=torch.float32)
                input_mask = torch.zeros((batch_size, 128), dtype=torch.int32)
                
                print("Processing each sample in batch...", flush=True)
                for i, row in enumerate(x):
                    try:
                        atom_idx = int(torch.FloatTensor(row)[70401].item()) if row.shape[0] > 70401 else 0
                        input_mask[i, atom_idx] = 1
                        input[i, atom_idx] = float(y[i].item())
                    except IndexError as e:
                        print(f"Index error: {e}, row.shape={row.shape}", flush=True)
                        atom_idx = 0
                        input_mask[i, atom_idx] = 1
                        input[i, atom_idx] = float(y[i].item())

                print("Calculating metrics...", flush=True)
                try:
                    # Check and fix output tensor dimensions
                    print(f"Output original shape: {output.shape}", flush=True)
                    if output.dim() > 2:  # If output has more than 2 dimensions
                        output = output.squeeze(-1)  # Squeeze last dimension
                        print(f"Output squeezed shape: {output.shape}", flush=True)
                    
                    # Compare input and output tensors
                    print(f"Comparing shapes - Output: {output.shape}, Input: {input.shape}", flush=True)
                    
                    batch_mse = ((output - input) ** 2).mean().item()
                    batch_mae = torch.abs(output - input).mean().item()
                    meanSquaredError += batch_mse * batch_size
                    meanAverageError += batch_mae * batch_size
                    processed_batches += 1
                    print(f"Batch {processed_batches} metrics: MSE={batch_mse}, MAE={batch_mae}", flush=True)
                except Exception as e:
                    print(f"Error calculating metrics: {e}", flush=True)
                    print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'unknown'}", flush=True)
                    print(f"Input shape: {input.shape}", flush=True)
                    raise

            if total_samples > 0:
                meanAverageError /= total_samples
                meanSquaredError /= total_samples
            else:
                print("Warning: No samples processed", flush=True)
                meanAverageError = float('inf')
                meanSquaredError = float('inf')
        
        return float(meanAverageError), float(meanSquaredError)

    print("Loading test dataset...", flush=True)
    testset = read_data(data_path)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=settings['batch_size'], shuffle=True)

    try:
        print("Starting evaluation...", flush=True)
        test_mae, test_mse = evaluate(model, loss, test_loader)
        print(f"MAE: {test_mae:.6f}, MSE: {test_mse:.6f}", flush=True)
    except Exception as e:
        print(f"failed to validate the model {e}", flush=True)
        test_mae = float('inf')
        test_mse = float('inf')
    
    print("Preparing validation report...", flush=True)
    # Create a report with scalar values to be displayed in FEDn Studio UI
    report = { 
                "classification_report": 'unevaluated',
                "MAE": test_mae,
                "MSE": test_mse,
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

if __name__ == '__main__':
    print("Checking command line arguments...", flush=True)
    if len(sys.argv) < 3:
        print("Usage: python validate.py <model_path> <output_path> [data_path]", flush=True)
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print("Determining data path...", flush=True)
    data_path = os.environ.get('FEDN_TEST_CSV')
    if not data_path and len(sys.argv) > 3:
        data_path = sys.argv[3]
    
    if not data_path or not os.path.exists(data_path):
        if data_path:
            print(f"WARNING: Specified file not found: {data_path}", flush=True)
        
        if not data_path or not os.path.exists(data_path):
            print(f"WARNING: No data file found. Please specify the correct path to CSV file.", flush=True)
            print(f"Either set the FEDN_TEST_CSV environment variable or provide the CSV path as a command line parameter.", flush=True)
            sys.exit(1)
    
    print(f"Model path: {model_path}", flush=True)
    print(f"Output path: {output_path}", flush=True)
    print(f"Data path: {data_path}", flush=True)

    print("Loading settings...", flush=True)
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            print(f"Settings file reading error: {e}", flush=True)
            raise(e)

    print("Initializing FEDn helper...", flush=True)
    helper = get_helper(HELPER_MODULE)
    
    from models.pytorch_model import create_seed_model
    
    print("Creating model...", flush=True)
    model, loss, optimizer = create_seed_model(settings)
    print('Model created, starting evaluation', flush=True)
    
    print("Loading model weights...", flush=True)
    try:
        model_weights = helper.load(model_path)
        print(f"Loaded weights type: {type(model_weights)}", flush=True)
        if isinstance(model_weights, list):
            print(f"Model weights is a list with {len(model_weights)} elements", flush=True)
        elif isinstance(model_weights, dict):
            print(f"Model weights is a dict with {len(model_weights.keys())} keys", flush=True)
        
        # Pass current model to conversion function
        weights_dict = np_to_weights(model_weights, model)
        model.load_state_dict(weights_dict)
        print("Model successfully loaded", flush=True)
    except Exception as e:
        print(f"Model loading error: {e}", flush=True)
        print("WARNING: Model could not be loaded, evaluation may fail", flush=True)
        # Normal exit is more appropriate for FEDn in this case
        # Because validation cannot be performed if model cannot be loaded
        report = {
            "classification_report": 'failed',
            "MAE": float('inf'), 
            "MSE": float('inf'),
            "error": str(e)
        }
        # Use FEDn helper function
        save_metrics(report, output_path)
        sys.exit(1)

    print("Running validation...", flush=True)
    report = validate(model, data_path, loss, settings)

    print("Saving evaluation results...", flush=True)
    try:
        # Use FEDn helper function
        save_metrics(report, output_path)
        print(f"Evaluation report saved: {output_path}", flush=True)
    except Exception as e:
        print(f"Report saving error: {e}", flush=True)
        with open(output_path, "w") as fh:
            fh.write(json.dumps(report))
        print(f"Fallback: Saved report using direct JSON write", flush=True)

