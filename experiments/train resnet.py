import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from readData import read_dataset
from load_ImageNet import *
from ResNet import ResNet18, ResNet101, ResNet50
from ATJU import ATJU
from torch.optim import lr_scheduler
import time
import os



def save_txt(rewards, file_name, txt_name):
    rewards = np.atleast_1d(rewards).flatten()
    # Create directory if it doesn't exist
    dir_path = os.path.join('./txt_save', file_name)
    os.makedirs(dir_path, exist_ok=True)
    # Full file path
    file_path = os.path.join('./txt_save/'+file_name, f'{txt_name}.txt')
    # Append mode; open(..., 'a') auto-creates the file if it doesn't exist
    with open(file_path, 'a', encoding='utf-8') as f:
        # Write line by line for easier reading
        for r in rewards:
            f.write(f'{r}\n')


def main():

    now_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load data
    batch_size = 256
    # train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path='dataset')
    train_loader, valid_loader = load_ImageNet(ImageNet_PATH=r"C:\Users\Administrator\Desktop\ATJU\experiments\imagenet",
                                               batch_size=batch_size, workers=6, pin_memory=True)
    # Load model (use pretrained model, modify last layer, freeze previous weights)
    n_class = 1000
    model = ResNet50(num_classes=n_class)
    """
    The 7x7 downsampling convolution and pooling in ResNet18 can lose information,
    so we replace the 7x7 conv and max pooling with a 3x3 downsampling conv,
    and reduce the stride and padding size accordingly.
    """
    # model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.fc = torch.nn.Linear(512, n_class)  # Replace the last fully connected layer
    model = model.to(device)
    # Use cross-entropy loss function
    criterion = nn.CrossEntropyLoss().to(device)

    n_epochs = 150
    valid_loss_min = np.inf  # track change in validation loss
    accuracy = []

    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), weight_decay=1e-4, eps=1e-8)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    # Start training



    for epoch in tqdm(range(1, n_epochs+1)):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0

        ###################
        # Train the model #
        ###################
        model.train() # Enable batch normalization and dropout
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data).to(device)  # Equivalent to output = model.forward(data).to(device)
            # Compute loss
            loss = criterion(output, target)
            # Backward pass: compute gradient of the loss w.r.t. model parameters
            loss.backward()
            # Perform a single optimization step (parameter update)
            optimizer.step()
            # Update loss
            train_loss += loss.item()*data.size(0)

        ######################
        # Validate the model #
        ######################

        model.eval()  # Evaluation mode
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data).to(device)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item()*data.size(0)
                # Convert output probabilities to predicted class
                _, pred = torch.max(output, 1)
                # Compare predictions with ground truth labels
                correct_tensor = pred.eq(target.data.view_as(pred))
                # correct = np.squeeze(correct_tensor.to(device).numpy())
                # total_sample += batch_size

                total_sample += target.size(0)
                right_sample += correct_tensor.sum().item()

            print("Accuracy:",100*right_sample/total_sample,"%")
            accuracy.append(right_sample/total_sample)

            # Compute average loss
            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)

            # Display training and validation loss
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # Save model if validation loss decreases
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
                torch.save(model.state_dict(), 'model_dict/resnet50.pt')
                valid_loss_min = valid_loss

            save_txt(valid_loss, file_name=f'imagenet_valid_loss', txt_name=f'SGD_{now_time}')
            save_txt(train_loss, file_name=f'imagenet_train_loss', txt_name=f'SGD_{now_time}')
            save_txt(right_sample / total_sample, file_name=f'imagenet_valid_acc', txt_name=f'SGD_{now_time}')

            scheduler.step()

            print(f"Current LR: {optimizer.param_groups[0]['lr']}")  # Display current learning rate


if __name__ == '__main__':
    main()