# # -*- coding: utf-8 -*-
# """
# Created on Tue Aug 24 08:52:52 2021

# @author: Admin
# """
# import torch, json
# import numpy as np
# from torchvision import datasets, transforms
# from PIL import Image
# import cv2
# import models.CDCNs as CDCNs
# import matplotlib.pyplot as plt

# test_image = "test2.png"

# # Prepare the labels
# #with open("labels.json") as f:
# #    labels = json.load(f)
# # First prepare the transformations: resize the image to what the model was trained on 
# #and convert it to a tensor
# data_transform = transforms.Compose([
#     transforms.Resize((256, 256)), 
#     transforms.ToTensor()
# ])
# #read image
# image = cv2.imread(test_image)
# # print(image)
# # image = cv2.resize(image,dsize= (256, 256))
# #print(image.shape)
# #transpose for the correct shape to feed into the next steps
# image = image.transpose(2,1,0)
# print(image.shape)
# #plt.imshow(image), plt.xticks([]), plt.yticks([])
# # Now apply the transformation, expand the batch dimension, and send the image to the GPU
# # image = data_transform(image).unsqueeze(0)
# image = torch.from_numpy(image).float().unsqueeze(0)
# #print(type(image))

# model = CDCNs.CDCNpp()#(pretrained=True)
# # Send the model to the GPU 
# # model.cuda()
# # Set layers such as dropout and batchnorm in evaluation mode
# model.eval()
# # Get the 1000-dimensional model output
# outputs = model(image)
# print(outputs[0].shape)

# #only the x and y axis are used since depth is only 1 
# #by taking the mean of the depth map, the result of genuineness is found       
# with torch.no_grad():
#         score = torch.mean(outputs[0], axis=(1,2))
#         print(score)
# #decide whether fake or genuine using score found   
# if score >= 0.6:
#     result = "genuine"
#     print("genuine")
# else:
#     result = "fake"
#     print("fake")

#from video photo, find face mtccn, draw square on it, then write the score on top
from mtcnn import MTCNN
import torch, json
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import cv2
import models.CDCNs as CDCNs
import matplotlib.pyplot as plt
from utils.utils import read_cfg, get_optimizer, get_device, build_network
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#path to the trained model
PATH = "experiments/CDCNpp_nuaa_360.pth"
# cap = cv2.VideoCapture("face.mp4")

#this section is for loading the model
model = CDCNs.CDCNpp()
model_main = torch.load(PATH, map_location=torch.device('cpu'))
#then using load_state_dict the parameters are loaded, strict is added 
#to prevent the error because of deleting part of the model
model.load_state_dict(model_main['state_dict'], strict=False)
#cuda is not necessary and eval makes it worse so commented
#model.cuda()
model.eval()

test_image = "test2.png"
img = cv2.imread(test_image)
detector = MTCNN()
#change colour space to rgb for the mtcnn
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#finally the resulting faces properties are assigned to a variable 
result = detector.detect_faces(image_rgb)

#if there is a face present to the following:
if len(result) > 0:
    #extract keypoint and bounding box information from the result variable
    keypoints = result[0]['keypoints']
    bounding_box = result[0]['box']
    
    #assign bounding box infromation to proper variables for ease of use                    
    x1 = bounding_box[0]
    y1 = bounding_box[1]
    x2 = bounding_box[0]+bounding_box[2]
    y2 = bounding_box[1] + bounding_box[3]  
    
    #draw a rectangle on the picture to show the faces place          
    cv2.rectangle(img, (x1, y1), (x2, y2),(50,255,0),2)


#crop the image to include only the face section
image_cr = img[y1:y2,x1:x2]

#transform the face and transpose for it to be usable in the from_numpy
data_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
image = image_cr.transpose(2,1,0)
# Now apply the transformation, expand the batch dimension, and send the image to the GPU
image = torch.from_numpy(image).float().unsqueeze(0)#.cuda(0)

#after the image is in the proper form, it is loaded to the CDCNN model for output
outputs = model(image)

#torch.no_grad() is used to prevent errors from clashes 
#and only output's [0] tuple part used out of the 6 since that 
#part is the depth map which is the wanted output.
with torch.no_grad():
        #only the x and y axis are used since depth is only 1 
        #by taking the mean of the depth map, the result of genuineness is found
        score = torch.mean(outputs[0], axis=(1,2))
        #for finding the pixel number
        #score = torch.sum(outputs[0] > 0.05) 
        print(score)
        
#if the resulting score is bigger than 0.6 it is genuine otherwise it is fake     
if score >= 0.2: # or 150 if going from pixel number
    result = "genuine"
    print("genuine")
else:
    result = "fake"
    print("fake")
    
#to print genuine or not at the bottom of the window
cv2.putText(img, str(result), (x2 - 85, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.putText(img, str(round(float(score),4)), (x2-85, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

#this parts are for showing the rseults, the image and the depth map
cv2.imshow("mtcnn face",img)
plt.imshow(img)
cv2.imwrite('mtcnn face.jpg',img)
cv2.destroyAllWindows()