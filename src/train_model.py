import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import preprocess_data
from sklearn.metrics import classification_report
import pickle

# load preprocessed data
X_train, X_test, y_train, y_test = preprocess_data()

# convert to PyTorch tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# define neural network
class TriageNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_classes=3):
        super(TriageNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

input_size = X_train.shape[1] # number of features
# initialize model, neural network instance
model = TriageNN(input_size)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# evaluate
with torch.no_grad():
    y_pred = model(X_test_tensor) 
    y_pred_classes = torch.argmax(y_pred, axis=1) # pick class with highest score
    print(classification_report(y_test_tensor, y_pred_classes))

# save model - save only the weights
torch.save(model.state_dict(), "../models/triage_nn_model.pth")