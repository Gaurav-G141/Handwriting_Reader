import torch
import random


torch.set_default_dtype(torch.float16)

#read csv file
print("Opening")
file = open("A_Z Handwritten Data.csv","r")
print("Parsing")
lines = file.readlines()
print("Tensoring")
all_tensors = []
for line in lines:
    if (random.randint(1,10000) == 10000):
        break
    #Convert each line to a tensor
    s = line.replace("\n","")
    s = s.split(",")
    s = [int(i) for i in s]
    tensor = (torch.tensor(s[1:]))
    ans_tensor = torch.zeros(1, 26)
    ans_tensor[0, s[0]] = 1
    print(ans_tensor)
    all_tensors.append(tensor)
    all_tensors.append(ans_tensor)
torch.save(all_tensors, "smalldata.pt")


