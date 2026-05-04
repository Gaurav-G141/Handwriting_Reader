import torch
import random
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
test_split = 0.2              # fraction of pairs used for testing (e.g. 0.2 = 20%)
seed = random.randint(1,10000) # random seed for reproducibility (set to None to disable)
tensors = list(torch.load("smalldata.pt", weights_only=True))
tensors = [t.to(torch.float16) for t in tensors]
torch.set_default_dtype(torch.float16)

# Creates the list of tensors as needed
pairs = [(tensors[i], tensors[i + 1]) for i in range(0, len(tensors), 2)]

if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
random.shuffle(pairs)

n_test  = max(1, round(len(pairs) * test_split))
n_train = len(pairs) - n_test
train_pairs = pairs[:n_train]
test_pairs  = pairs[n_train:]
print(f"Split → {n_train} training pairs | {n_test} test pairs  "
      f"({100*(1-test_split):.0f}% / {100*test_split:.0f}%)")

data_train, ans_train = zip(*train_pairs) if train_pairs else ([], [])
data_test,  ans_test  = zip(*test_pairs)  if test_pairs  else ([], [])




#img = data_train[500].detach().cpu().float()
#img = img.view(28, 28)
#plt.figure(figsize=(3, 3))
#plt.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
#plt.axis("off")
#plt.tight_layout()
#plt.show()

hidden_1 = 10
hidden_2 = 15
#The actual model
class Model(nn.Module):
    #Input layer
    #Hidden 1
    #Hidden 2
    #Output
    def __init__(self, in_feats = 784, h1 = hidden_1, h2 = hidden_2, out_feats=26):
       super().__init__()
       self.fc1 = nn.Linear(in_features=in_feats, out_features=h1)
       self.fc2 = nn.Linear(in_features=h1, out_features=h2)
       self.fc3 = nn.Linear(in_features=h2, out_features=out_feats)

    #Forwarding
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#Initalize model
torch.manual_seed(random.randint(1,1000))
model = Model()
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.001)

for epochs in range(100):
    loss = 0
    for i in range(len(data_train)):

        pred = model.forward(data_train[i])
        print(pred)
        loss += crit(pred, ans_train[i])
    print(f"Total loss for this run: {loss}")