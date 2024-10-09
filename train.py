import time
import copy
import torch

# Early stopping function
def early_stopping(val_loss, best_val_loss, patience_counter, patience=5, delta=0):
    """
    Args:
        val_loss (float): Current validation loss.
        best_val_loss (float): Best validation loss observed so far.
        patience_counter (int): Number of epochs since last improvement.
        patience (int): How long to wait after last time validation loss improved.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.

    Returns:
        bool: Whether to stop training.
        float: Updated best validation loss.
        int: Updated patience counter.
    """
    if best_val_loss - val_loss > delta:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        return True, best_val_loss, patience_counter
    else:
        return False, best_val_loss, patience_counter

# Training loop with early stopping function
def train_model(model, dataloaders, device, criterion, optimizer, scheduler, max_epochs=25, patience=5):
    """
    Train the model with early stopping.
    
    Args:
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Model optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs.
        patience (int): How long to wait after last time validation loss improved.
        
    Returns:
        torch.nn.Module: Trained model.
        dict: Training and validation losses.
    """
    since = time.time()
    losses = {'train': [], 'val': []}

    best_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        print(f'Epoch {epoch}/{max_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            losses[phase].append(epoch_loss)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Check early stopping only in validation phase
            if phase == 'val':
                stop, best_val_loss, patience_counter = early_stopping(epoch_loss, best_val_loss, patience_counter, patience=patience)
                if epoch_loss > best_val_loss:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # Early stopping check
        if stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, losses