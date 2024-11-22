import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
my_device = 'mps'

class Model(nn.Module):
    def __init__(self, take_in=4, h1=8, h2=16, h3=32, h4=16, h5=8, out_put=3):
        super().__init__()
        self.fc1 = nn.Linear(take_in, h1, device=my_device, dtype=torch.float)
        self.fc2 = nn.Linear(h1, h2, device=my_device, dtype=torch.float)
        self.fc3 = nn.Linear(h2, h3, device=my_device, dtype=torch.float)
        self.fc4 = nn.Linear(h3, h4, device=my_device, dtype=torch.float)
        self.fc5 = nn.Linear(h4, h5, device=my_device, dtype=torch.float)
        self.fc6 = nn.Linear(h5, out_put, device=my_device, dtype=torch.float)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


torch.manual_seed(21)
model = Model()

df = pd.read_csv("iris.csv")

df['variety'] = df['variety'].replace('Setosa', 0.0)
df['variety'] = df['variety'].replace('Versicolor', 1.0)
df['variety'] = df['variety'].replace('Virginica', 2.0)
y = df['variety'].values
x = df.drop('variety', axis=1).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=4)

x_train = torch.tensor(x_train, device=my_device, dtype=torch.float)
x_test = torch.tensor(x_test, device=my_device, dtype=torch.float)
y_train = torch.tensor(y_train, device=my_device, dtype=torch.long)
y_test = torch.tensor(y_test, device=my_device, dtype=torch.long)

criteria = nn.CrossEntropyLoss()
optimal = torch.optim.Adam(model.parameters(), lr=0.1)

trials = 1000
losses = []
for i in range(trials):
    predict = model.forward(x_train)
    loss = criteria(predict, y_train)
    losses.append(loss.detach().to('cpu').numpy())
    optimal.zero_grad()
    loss.backward()
    optimal.step()
    if i % 10 ==0:
        print(f'Step{i}, loss{loss}')

plt.plot(range(trials), losses)
plt.show()

