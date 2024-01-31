import numpy as np
import torch
import fusecam
import einops
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

def train_volume_and_slice(net, 
                           loss_function, 
                           optimizer, 
                           dataloader, 
                           num_epochs, 
                           interpolate_function, 
                           device='cuda:0'):
    """
    Train a neural network on 3D and corresponding 2D image data.

    Parameters:
    net (torch.nn.Module): The neural network model to be trained.
    loss_function (callable): The loss function used for training.
    optimizer (torch.optim.Optimizer): Optimizer used for training.
    dataloader (torch.utils.data.DataLoader): DataLoader containing the training data.
    num_epochs (int): Number of epochs to train the model.
    interpolate_function (callable): Function used to interpolate 3D images to 2D.
    device (str, optional): The device (CPU/GPU) on which to perform training. Defaults to 'cuda:0'.

    Returns:
    None. The function trains the model in-place and prints the training loss.
    """
    net.to(device)  # Move the network to the specified device
    net.train()  # Set the network to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in dataloader:
            img_tensor_3d, flat_2d_tensor, weights, indices = [item.to(device) for item in batch]

            # Forward pass
            outputs = net(img_tensor_3d)
            loss = 0.0
            for img3d, img2d, ws, idx in zip(outputs, flat_2d_tensor, weights, indices):
                img_flat = img3d.flatten()
                interp = interpolate_function(img_flat, idx, ws)
                not_nan_sel = ~torch.isnan(interp)

                # Compute loss
                loss += loss_function(interp[not_nan_sel], img2d[not_nan_sel])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")


