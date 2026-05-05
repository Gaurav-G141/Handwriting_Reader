import torch
import random
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
test_split = 0.2              # fraction of pairs used for testing (e.g. 0.2 = 20%)
seed = random.randint(1,10000) # random seed for reproducibility (set to None to disable)
print("Loading")
tensors = list(torch.load("data.pt", weights_only=True))
tensors = [t.to(torch.float32) for t in tensors]
tensors = [tensors[i][0] if i % 2 == 1 else tensors[i] for i in range(len(tensors))] #Every ans tenssor is 1D
torch.set_default_dtype(torch.float32)
print("Making pairs")
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

hidden_1 = 150
hidden_2 = 100
hidden_3 = 50
#The actual model
class Model(nn.Module):
    #Input layer
    #Hidden 1
    #Hidden 2
    #Output
    def __init__(self, in_feats = 784, h1 = hidden_1, h2 = hidden_2, h3 = hidden_3, out_feats=26):
       super().__init__()
       self.fc1 = nn.Linear(in_features=in_feats, out_features=h1)
       self.fc2 = nn.Linear(in_features=h1, out_features=h2)
       self.fc3 = nn.Linear(in_features=h2, out_features=h3)
       self.fc4 = nn.Linear(in_features=h3, out_features=out_feats)

    #Forwarding
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return x


#Initalize model
torch.manual_seed(random.randint(1,1000))
model = Model()
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.0001)

for epochs in range(4):
    num_loss = 0
    print(f"Epoch: {epochs + 1}. ", end = "")
    for i in range(len(data_train)):
        pred = model.forward(data_train[i])
        loss = crit(pred, ans_train[i])
        num_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Total loss for this run: {num_loss}. ", end = "")
    correct = 0
    with torch.no_grad():
        for i in range(len(data_test)):
            pred = model.forward(data_test[i])
            if torch.argmax(pred).item() == torch.argmax(ans_test[i]).item():
                correct += 1
    print(f"Test accuracy: {correct}/{len(data_test)} ({100 * correct / len(data_test):.1f}%)")
torch.save(model.state_dict(), 'model.pth')