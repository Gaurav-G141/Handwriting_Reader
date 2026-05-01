import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

#read csv file
print("Opening")
file = open("A_Z Handwritten Data.csv","r")
print("Parsing")
lines = file.readlines()
print("Tensoring")
all_tensors = []
for line in lines:
    s = line.replace("\n","")
    s = s.split(",")
    s = [int(i) for i in s]
    tensor = (torch.tensor(s[1:]))
    ans_tensor = torch.zeros(1, 26)
    ans_tensor[0, s[0]] = 1
    print(ans_tensor)
    all_tensors.append(tensor)
    all_tensors.append(ans_tensor)
torch.save(all_tensors, "data.pt")


