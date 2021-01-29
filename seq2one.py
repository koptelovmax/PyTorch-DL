# Hochreiter & Schmidhuber classification task (1997):
# X, X -> Q
# X, Y -> R
# Y, X -> S
# Y, Y -> U

import torch
import torch.nn as nn

from res.sequential_tasks import TemporalOrderExp6aSequence as QRSU
#%%
# Define the RNN class:
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.f = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.rnn(x)[0] # output the features
        x = self.f(x)
        return x
#%%
# Define a model and the task:

# Define difficulty of the task:        
difficulty = QRSU.DifficultyLevel.NORMAL # EASY, NORMAL, MODERATE or HARD
batch_size = 32

# Create data generators:
train_data = QRSU.get_predefined_generator(difficulty, batch_size)
test_data  = QRSU.get_predefined_generator(difficulty, batch_size)
    
# Define model parameters:
input_size  = train_data.n_symbols
hidden_size = 64 #4
output_size = train_data.n_classes

# Training on GPU:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define a model:
net = RNN(input_size, hidden_size, output_size).to(device)

# Optimizer and an objective function:
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3)

# An objective function:
criterion = torch.nn.CrossEntropyLoss()

# Additional parameters:
epochs = 10
#%%
# Train the network:

for epoch in range(epochs):
    
    # Number of correctly classified sequences:
    num_correct = 0
    
    for batch_id in range(len(train_data)):
        
        # Unpack the data:
        data_X, data_Y = train_data[batch_id]
        
        # Convert the data into tensors and send to GPU:
        data_X = torch.from_numpy(data_X).float().to(device)
        data_Y = torch.from_numpy(data_Y).long().to(device)
        
        # Compute the output:
        output = net(data_X)
        
        # Remove padding:
        output = output[:, -1, :]
        
        # Determine real labels for batches:
        data_Y = data_Y.argmax(dim=1)
        
        # Compute training loss:
        loss = criterion(output, data_Y)
        
        # Reset the gradients:
        optimizer.zero_grad()
        
        # Compute the gradients:
        loss.backward()
        
        # Update the parameters:
        optimizer.step()
        
        # Determine predicted labels for batches:
        pred_Y = output.argmax(dim=1)
        
        # Update the number of correctly classified sequences:
        num_correct += (pred_Y == data_Y).sum().item()
                
        # Display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss.item()))

print('Finished Training')
#%%
# Test the network on the test data:

# Number of correctly classified sequences:
num_correct = 0

# Disable gradient calculations during inference:
with torch.no_grad():
    for batch_id in range(len(test_data)):
        
        # Unpack the data:
        data_X, data_Y = test_data[batch_id]
        
        # Convert the data into tensors and send to GPU:
        data_X = torch.from_numpy(data_X).float().to(device)
        data_Y = torch.from_numpy(data_Y).long().to(device)

        # Compute predictions:
        output = net(data_X)
        
        # Remove padding:
        output = output[:, -1, :]

        # Determine real labels for batches:
        data_Y = data_Y.argmax(dim=1)
        
        # Compute test loss:
        loss = criterion(output, data_Y)

        # Determine predicted labels for batches:
        pred_Y = output.argmax(dim=1)
        
        # Update the number of correctly classified sequences:
        num_correct += (pred_Y == data_Y).sum().item()
                
        # Compute accuracy:
        accuracy = float(num_correct) / (len(test_data) * test_data.batch_size) * 100
        
# Display the testing loss:
print("loss = {:.6f}, accuracy = {:2.2f}%".format(loss.item(), accuracy))

print('Finished Testing')
#%%