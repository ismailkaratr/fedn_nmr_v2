from torch.utils.data import TensorDataset
import pandas as pd
import torch
import numpy as np

def read_data(data_path):
    """ Helper function to read and preprocess data from CSV files for training. """
    
    # Read CSV file
    pkd = np.array(pd.read_csv(data_path))
    X = pkd[:, :-1]
    y = pkd[:, -1:]

    # Adjust data types
    X = X.astype('float32')
    y = y.astype('float32')

    # Make necessary formatting
    X = np.expand_dims(X, 1)
    
    # Normalize input (scale 0-1)
    # Note: Removed the division by 255 as it might not be appropriate for NMR data
    # Add it back if your data needs this normalization
    # X /= 255
    
    tensor_x = torch.Tensor(X)  # Convert to PyTorch tensor
    tensor_y = torch.from_numpy(y)
    dataset = TensorDataset(tensor_x, tensor_y)  # Create dataset

    return dataset