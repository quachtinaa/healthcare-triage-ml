import torch
from preprocess import preprocess_data
from train_model import TriageNN
from sklearn.metrics import classification_report

# ------------------------------
# Step 1: Load preprocessed data
# ------------------------------
X_train, X_test, y_train, y_test = preprocess_data()

# ------------------------------
# Step 2: Convert test data to PyTorch tensor
# ------------------------------
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ------------------------------
# Step 3: Initialize model and load trained weights
# ------------------------------
input_size = X_train.shape[1]
model = TriageNN(input_size)
model.load_state_dict(torch.load("../models/triage_nn_model.pth"))
model.eval()  # important: set model to evaluation mode

# ------------------------------
# Step 4: Predict on test set (simulate new patients)
# ------------------------------
with torch.no_grad():  # no gradients needed
    y_pred = model(X_test_tensor)
    y_pred_classes = torch.argmax(y_pred, axis=1)

# ------------------------------
# Step 5: Show predictions
# ------------------------------
print("Predictions for test set (simulated new patients):")
print(y_pred_classes)

# ------------------------------
# Step 6: Evaluate model performance
# ------------------------------
print("\nClassification report on test set:")
print(classification_report(y_test_tensor, y_pred_classes))