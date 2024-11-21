import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

my_devive = torch.device("mps")
class Model(nn.Module):
    # Input Layer(4 feature of a flower to) -> hidden layer1 -> h2(n)
    # -> output 3 layers
    def __init__(self, in_feature=4, h1=8, h2=9, output_features=3, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_feature, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        return x

torch.manual_seed(41)
model = Model()
my_df = pd.read_csv("iris.csv")
print(my_df)
# need to replace the string with num, since nums better for formula
my_df["variety"] = my_df['variety'].replace("Setosa", 0.0)
my_df["variety"] = my_df['variety'].replace("Versicolor", 0.0)
my_df["variety"] = my_df['variety'].replace("Virginica", 0.0)
print(my_df)

# Start training
y = my_df["variety"]
x = my_df.drop('variety', axis=1)
x = x.values
y = y.values
print(x)

from sklearn.model_selection import train_test_split

# run train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set criterion of model to measure the error
cirterion = nn.CrossEntropyLoss()
# choose Adam Optimizer Lr = Learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Trainmodel!
epochs = 100
losses = []
for i in range(epochs):
    # go forward
    y_pred = model.forward(x_train)
    # Measure the loss/error
    loss = cirterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
    if i % 10 == 0:
        print(f'epoch:{i}, and loss:{loss}')

# do some back propagation Take error rate of forward propagation and feed it
    # back thru the network to fine tune the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("epoch")
plt.show()
