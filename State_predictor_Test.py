import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import State_predictor
import torch.utils.data as Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = np.random.rand(500)*30
y = np.sin(0.01*x)

#plt.scatter(x, y)
#plt.show()
x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1).to(device)  # x data (tensor), shape=(100, 1)
y = torch.sin(4*x).to(device) + 0.5*torch.rand(x.size()).to(device) 

BATCH_SIZE = 64
EPOCH = 5000

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True)
#print(x, y)
net = State_predictor(1,1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()


inputs = Variable(x)
outputs = Variable(y)
for epoch in range(EPOCH):

    for step, (batch_x, batch_y) in enumerate(loader):

        bx = Variable(batch_x)
        by = Variable(batch_y)
        prediction = net(batch_x)
        loss = loss_func(prediction, by) 
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()       

    #if epoch % 10 == 0:
    #    # plot and show learning process
    #    plt.cla()
    #    plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
    #    plt.plot(bx.data.cpu().numpy(), prediction.cpu().data.numpy(), 'r*', lw=2)
    #    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.cpu().numpy(), fontdict={'size': 10, 'color':  'red'})
    #    plt.pause(0.1)

prediction = net(inputs)
plt.plot(x.data.cpu().numpy(), prediction.cpu().data.numpy(), 'r*', lw=2)
plt.show()