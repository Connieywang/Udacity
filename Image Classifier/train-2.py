
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import seaborn as sns 

parser = argparse.ArgumentParser(description='Training Your Network')

parser.add_argument ('--data_dir', action = 'store', type = str, default="./flowers/", help = 'Locate a data directory')

parser.add_argument('--arch', action='store', type = str, default = 'vgg16', help= 'Provide a pretrained model. Only VGG family model is supported. The default model is Vgg16')

parser.add_argument('--save_dir', action = 'store', type = str, default = 'checkpoint.pth', help = 'Locate a saving directory')

parser.add_argument('--lr', action = 'store',type=float, default = 0.001, help = 'learning rate, default value is 0.001')

parser.add_argument('--dropout', action = 'store', type=float, default = 0.05, help = 'Dropout for training, default value is 0.05')

parser.add_argument('--hidden_units', action = 'store', type=int, default = 4096, help = 'Number of hidden units in classifier, default value is 4096')

parser.add_argument('--epochs', action = 'store', type = int, default = 5, help = 'Number of epochs for traning')

parser.add_argument('--gpu', action = "store", default = True, help = 'Turn on or off gpu mode')

parser.add_argument('--output', action = "store", type = int, default = 102, help = 'Provide the output size. The default size is 102')


results, _ = parser.parse_known_args()

save_dir = results.save_dir
data_dir = results.data_dir
learning_rate = results.lr
dropout = results.dropout
hidden_units = results.hidden_units
epochs = results.epochs
gpu = results.gpu
pre_model = results.arch
output_size = results.output

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
                    
#data loading 
def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50) 
    
    return train_data, test_data, valid_data, trainloader, testloader, validloader

train_data, test_data, valid_data, trainloader, testloader, validloader = load_data(data_dir)  

#Build model 
if pre_model.startswith('vgg'):
    model = getattr(models, pre_model)(pretrained=True)
else:
    print("The supplied model not supported. Defaulting to VGG16.")
    model = getattr(models, 'vgg16')(pretrained=True)

                    
def load_classifier (model, hidden_units, dropout,output_size):
                                                          
        for param in model.parameters():
            param.requires_grad = False
       
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.classifier = classifier
        
        return model 

load_classifier(model, hidden_units, dropout, output_size)
                    
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
                    

def validation(model,validloader, criterion, gpu):
    valid_loss = 0
    accuracy = 0  
    
    for ii, (inputs, labels) in enumerate(validloader):
        
        
        if gpu == True:
        # Move input and label tensors to the GPU
            if torch.cuda.is_available():
                device = torch.device('cuda')
                inputs, labels = inputs.to(device), labels.to(device)
            else:
                device = torch.device('cpu')
                inputs, labels = inputs.to(device), labels.to(device)
        else:
            device = torch.device('cpu')
            inputs, labels = inputs.to(device), labels.to(device)
            
        outputs = model.forward(inputs)
        valid_loss = criterion(outputs, labels).item()
        
        #Caculate Probability
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1,dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    return valid_loss, accuracy 
                    
#Train Model                    
def train_model(model, epochs,trainloader, validloader, criterion, optimizer,gpu):                    
    steps = 0
    running_loss = 0
    print_every = 30 #print out validation loss 
    
    if gpu == True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model.to(device)
        else:
            print('Cuda is not available, use CPU')
            device = torch.device('cpu')
            model.to(device)    
    else:
        device = torch.device('cpu')
        model.to(device)
        

    for epoch in range(epochs):
    
        for inputs, labels in trainloader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
                                
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
            
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu)
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {valid_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")   #test and valid loader the same size?
            
                running_loss = 0
                model.train()
                
    return model, optimizer

model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer,gpu)

#Test model                    
def check_accuracy_on_test(testloader,gpu):
    correct = 0
    total = 0
    model.eval() 
    
    with torch.no_grad():
        
        for data in testloader:
                                        
            images, labels = data
            
            if gpu == True:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    images, labels = images.to(device), labels.to(device)
        
                else:
                    device = torch.device('cpu')
                    images, labels = images.to(device), labels.to(device)
            else:
                device = torch.device('cpu')
                images, labels = images.to(device), labels.to(device)   

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
check_accuracy_on_test(testloader,gpu)             
        

model.class_to_idx = train_data.class_to_idx

checkpoint = {'epochs_num': epochs,
              'arch': pre_model,
              'hidden_units': hidden_units,
              'output_size' : output_size,
              'dropout' : dropout,
              'optimizer_state': optimizer.state_dict,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}

torch.save(checkpoint, save_dir)                    
                    
                    
                    
                    
                    