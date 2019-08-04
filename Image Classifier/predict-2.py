
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

parser = argparse.ArgumentParser(description='Predict the Image by Network')
                    
parser.add_argument('--image_dir', action='store', default = "flowers/test/10/image_07090.jpg",type = str, help='Locate the path for image')

parser.add_argument('--load_dir', action='store', default = 'checkpoint.pth', help='Provide Checkpoint file')

parser.add_argument('--top_k', action='store', type=int, default = 5, help='Enter the top number of classes, the default value is 5')

parser.add_argument('--flower_name', action='store', help='Mapping classes to flower names')

parser.add_argument('--gpu', action = "store", default = False, help = 'Turn on or off gpu mode')

results, _ = parser.parse_known_args()                  

image_dir = results.image_dir
load_dir = results.load_dir
top_k = results.top_k
flower_name =results.flower_name
gpu = results.gpu


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#load checkpoint 
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    model = getattr(models,arch)(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    output_size = checkpoint['output_size']

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
     
    return model

model= load_checkpoint(load_dir)

#process image 
def process_image(image):
    
    image = Image.open(image)
    
    if image.width > image.height:
        image.thumbnail((99999, 256))
    else:
        image.thumbnail((256,99999))
    
    left = (image.width-224)/2
    bottom = (image.height-224)/2
    right = left + 224
    top = bottom + 224
    image = image.crop((left, bottom, right, top))
    
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    image = (image - mean)/std
    
    image = image.transpose((2, 0, 1))
    
    return image

processed_image = process_image(image_dir)

#predict image 
def predict(image_path, model, topk):

    img = process_image(image_path)
    
    if gpu == True:
        if torch.cuda.is_available():
            device = torch.device('cuda')            
            img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        else:
            print('Cuda is not available, use CPU')
            device = torch.device('cpu')
            img = torch.from_numpy(img).type(torch.FloatTensor) 
    else:
        device = torch.device('cpu')
        img = torch.from_numpy(img).type(torch.FloatTensor) 
        
    img = img.unsqueeze_(0)
    new_model = model.to(device)
    
    img.to(device)
    
    new_model.eval()
    
    with torch.no_grad():
        output = new_model.forward(img)
    
    probs=torch.exp(output)
   
    top_probs = probs.topk(topk)[0]
    top_labels = probs.topk(topk)[1]

    #convert to numpy array 
    probs_top_list = np.array(top_probs)[0]
    label_top_list = np.array(top_labels)[0]

    
    #give each class a index number 
    class_to_idx = new_model.class_to_idx

    # Inverting index-class dictionary
    indx_to_class = {key: val for val, key in class_to_idx.items()}
 
    
    # Converting index list to class list
    classes_top_list = []
    #find the class based on the index in the label_top_list 
    for index in label_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list

probability, classes = predict(image_dir, model,top_k)

print(probability)
print(classes)

flower_names = []
for i in classes:
    flower_names += [cat_to_name[i]]

print(f"This flower is most likely to be '{flower_names[0]}' with probability of {round(probability[0]*100,2)}% ")




