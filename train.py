#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from time import time
from get_input_parse import get_train_args
from model_utils import train_transformer, test_transformer, data_loader, check_value, get_criterion, get_optim, model_loader, get_device, validation, validate_model, initial_checkpoint

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
print_every = 30


def main():
    start_time = time()
    steps = 0
    in_arg = get_train_args()
    print(in_arg)
    trainloader = data_loader(train_transformer(train_dir))
    validloader = data_loader(test_transformer(valid_dir), train=False)
    testloader = data_loader(test_transformer(test_dir), train=False)
    learning_rate = check_value(int(in_arg.learning_rate), 0.001) 
    
    # Load Model
    model = model_loader(in_arg.arch, in_arg.hidden_units)
    criterion = get_criterion()
    try:
        optimizer = get_optim(model.classifier.parameters(), learning_rate)
    except:
        optimizer = get_optim(model.fc.parameters(), learning_rate)
        
    device = get_device(in_arg.gpu)
    model.to(device);

    # Train Model
    epochs = check_value(int(in_arg.epochs), 5)
    model.train()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
            
                running_loss = 0
                model.train()
    
    # Quickly Validate the model
    validate_model(model, testloader, device)
    
    # Save the model
    initial_checkpoint(model, in_arg.save_dir, train_data)
    
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", str(int((tot_time/3600))) + ":"+str(int((tot_time%3600)/60)) + ":" + str(int((tot_time%3600)%60)))

# Call to main function to run the program
if __name__ == "__main__":
    main()
