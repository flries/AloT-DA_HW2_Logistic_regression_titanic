# app.py - Titanic Survival Classification with PyTorch - Version 3

# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure kaggle module is installed: pip install kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Step 1: Business Understanding (CRISP-DM)
# The goal is to predict whether a passenger survived based on attributes like age, sex, and fare.

# Step 2: Data Understanding and Acquisition (CRISP-DM)
# Download data from Kaggle using Kaggle API

def download_data():
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files('titanic', path='./data')

    if not os.path.exists('./data/train.csv') or not os.path.exists('./data/test.csv'):
        os.system("unzip -o ./data/titanic.zip -d ./data")

download_data()

# Load the dataset
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/gender_submission.csv')

# Step 3: Data Preparation (CRISP-DM)
# Combine training and test datasets for consistent preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0
all_data = pd.concat([train_df, test_df], sort=False)

# Data Preprocessing
# Convert 'Sex' to numeric
all_data['Sex'] = LabelEncoder().fit_transform(all_data['Sex'])

# Age Binning
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
all_data['AgeGroup'] = pd.cut(all_data['Age'], bins=[0, 16, 80], labels=[0, 1])

# Simplify 'SibSp' into categories (0, 1, 2+)
all_data['SibSp'] = all_data['SibSp'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))

# Simplify 'Parch' into categories (0, 1, 2+)
all_data['Parch'] = all_data['Parch'].apply(lambda x: 0 if x == 0 else (1 if x == 1 else 2))

# Fare Binning
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['FareGroup'] = pd.cut(all_data['Fare'], bins=[0, 15, 30, 1000], labels=[0, 1, 2])

# Encode 'Embarked' as numeric categories
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
all_data['Embarked'] = LabelEncoder().fit_transform(all_data['Embarked'])

# Select features and target, explicitly dropping 'PassengerId'
features = ['Pclass', 'Sex', 'AgeGroup', 'SibSp', 'Parch', 'FareGroup', 'Embarked']
all_data = all_data[features + ['Survived', 'is_train', 'PassengerId']]

# Split back into train and test data, dropping PassengerId before scaling
train_data = all_data[all_data['is_train'] == 1].drop(['is_train', 'PassengerId'], axis=1)
test_data = all_data[all_data['is_train'] == 0].drop(['is_train', 'Survived', 'PassengerId'], axis=1)

X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
test_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)

# Split into training and validation sets (80-20)
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Custom Dataset class
class TitanicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Data loaders
train_dataset = TitanicDataset(X_train, y_train)
val_dataset = TitanicDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 4: Modeling (CRISP-DM)
# Define neural network model
class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(len(features), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = TitanicModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Evaluation (CRISP-DM)
# Training loop with validation
best_accuracy = 0.0
for epoch in range(2000):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_predicted = (val_outputs >= 0.5).float()
        val_accuracy = accuracy_score(y_val, val_predicted)
        
        # Save best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_titanic_model.pth')
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/2000], Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")

# Load the best model for evaluation
model.load_state_dict(torch.load('best_titanic_model.pth'))
model.eval()

# Step 6: Model Evaluation (Confusion Matrix and Accuracy)
# Predict on validation set
with torch.no_grad():
    val_outputs = model(X_val)
    val_predicted = (val_outputs >= 0.5).float()
    val_accuracy = accuracy_score(y_val, val_predicted)
    conf_matrix = confusion_matrix(y_val, val_predicted)

print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix on Validation Set")
plt.show()

# Step 7: Deployment (Generate Predictions for Submission)
# Predict on test data
with torch.no_grad():
    test_predictions = (model(test_tensor) >= 0.5).int().view(-1).numpy()

# Prepare submission
submission['Survived'] = test_predictions
submission[['PassengerId', 'Survived']].to_csv('titanic_submission.csv', index=False)
print("Submission file created.")
