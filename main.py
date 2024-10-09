import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from evaluate import evaluate_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
import scienceplots
import numpy as np
from image_utils import load_image_list, get_image_paths, show_examples
from train import train_model
from collections import Counter

#seeds set for reproducibility
np.random.seed(21)
torch.manual_seed(21)
torch.cuda.manual_seed_all(21)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#Setup datasets
data_dir = os.path.join(os.getcwd(), 'data')
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])

train_img_list = load_image_list(os.path.join(os.getcwd(), 'train_set.txt'))
test_img_list = load_image_list(os.path.join(os.getcwd(), 'test_set.txt'))

image_paths = get_image_paths(full_dataset, data_dir)

#Filter the dataset indices based on the train/test image lists
train_indices = [i for i, path in enumerate(image_paths) if path in train_img_list]
test_indices = [i for i, path in enumerate(image_paths) if path in test_img_list]

# Create Subset datasets for training and testing
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# Perform a random split on the training dataset to create a validation set (20% validation, 80% training)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size  

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

val_dataset.dataset.transform = data_transforms['test']
test_dataset.dataset.transform = data_transforms['test']

# Create DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0),
    'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
}

# Get dataset sizes and class names
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
print('Train Size: {}\nValidation Size: {}\nTest Size: {}\n'.format(dataset_sizes['train'], dataset_sizes['val'], dataset_sizes['test']))
class_names = full_dataset.classes

show_examples(full_dataset, class_names) 

class_counter = Counter()
# Iterate through the training dataset to count items in each class
for _, label in train_dataset:
    class_counter[label] += 1

# Convert Counter to a dictionary
class_counts = dict(class_counter)

# Print the number of items in each class
print("Number of items in each class for training:")
for class_index, count in class_counts.items():
    print(f"{class_names[class_index]}: {count} items")

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\nRunning on device:', device)   

# Load the pretrained model
model_ft = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.DEFAULT)

# Modify the final layer
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

""" Delete this comment to train model
# Train the model with early stopping
model_ft, losses = train_model(model_ft, dataloaders, device ,criterion, optimizer_ft, exp_lr_scheduler, max_epochs=100, patience=10)

# Create a dictionary to save the model state and losses
save_dict = {
    'model_state_dict': model_ft.state_dict(),
    'losses': losses
}

# Save the dictionary
torch.save(save_dict, 'model_and_losses.pth')
print('Model and losses saved successfully!')
"""

# Load the saved model and losses
save_dict = torch.load('model_and_losses.pth', weights_only=True)
model_ft.load_state_dict(save_dict['model_state_dict'])
losses = save_dict['losses']
model_ft = model_ft.to(device)
model_ft.eval()

# Evaluate the model on the test set
print() # Print an empty line for better readability
print('Evaluating the model on the test set...')
test_acc, test_loss, all_labels, all_preds = evaluate_model(model_ft, dataloaders['test'], criterion, device)
print(f'Test Accuracy: {test_acc:.4f}')

f1 = f1_score(all_labels, all_preds, average='weighted')
print(f'F1 Score: {f1:.4f}')

# Calculate precision and recall for each class
precision = precision_score(all_labels, all_preds, average=None)
recall = recall_score(all_labels, all_preds, average=None)
f1 = f1_score(all_labels, all_preds, average=None)

# Print precision and recall for each class
for class_name, p, r, f in zip(class_names, precision, recall, f1):
    print(f'Class {class_name} - Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f:.4f}')

#plot training losses and confusion matrix
with plt.style.context(['science', 'bright', 'no-latex']):
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 6))
    plt.plot(losses['train'], label='Train Loss')
    plt.plot(losses['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.xlim(left=0)
    plt.savefig('training_losses.png', dpi=330)
    plt.show()

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=class_names)
    # Plot the confusion matrix
    _, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.get_cmap('Blues'), ax=ax, colorbar=False)
    plt.savefig('confusion_matrix.png', dpi=330)
    plt.show()