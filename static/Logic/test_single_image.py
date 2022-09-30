
from model import Model
import numpy as np
import torch
from torchvision import transforms
import cv2


if __name__ == '__main__':
    
    image_path = 'test_subset/95.png' #Write the image path to be teested here!
    model_path = 'models/mymodel_cifar_0.35.pth.tar' #Write best trainig accuracy model here!   
    
    I = cv2.imread(image_path) 
    model = Model()
    model.load_state_dict(torch.load(model_path))  
    model.eval()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    classes = ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
    #(thresh, I1) = cv2.threshold(I, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    I1 = np.array(I)
    I1 = transform_test(I1)
    with torch.no_grad():
        I1 = torch.reshape(I1,(1,3,32,32))
        output = model(I1).detach()
        predict_y = np.argmax(output, axis=-1)
        answer = predict_y.numpy()[0]
        
        print("The object present in this image is: ",classes[answer])
