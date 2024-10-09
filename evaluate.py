import torch

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Test accuracy.
        float: Test loss.
        list: Ground truth labels.
        list: Predicted labels.
    """
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    return test_acc.item(), test_loss, all_labels, all_preds
