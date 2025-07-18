# Import libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

# Load CSVs manually
df_train = pd.read_csv("C:\\Users\\oshee\\Downloads\\sign_data\\sign_mnist_train.csv")
df_test = pd.read_csv("C:\\Users\\oshee\\Downloads\\sign_data\\sign_mnist_test.csv")


# Dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, dataframe):
        self.X = dataframe.drop('label', axis=1).values.reshape(-1, 1, 28, 28).astype('float32') / 255.0
        self.y = dataframe['label'].values.astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# PyTorch datasets and dataloaders
train_dataset = SignLanguageDataset(df_train)
test_dataset = SignLanguageDataset(df_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate model, loss, optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ART Classifier wrapper
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=25,
)

# Training loop
print("Training the model...")
for epoch in range(3):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {running_loss:.4f}")

# Prepare test data
x_test = np.array([img[0].numpy() for img in test_dataset])
y_test = np.array([img[1] for img in test_dataset])
y_test_oh = np.eye(25)[y_test]

# Adversarial attack (FGSM)
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Predict
preds_clean = classifier.predict(x_test)
preds_adv = classifier.predict(x_test_adv)

orig_preds = np.argmax(preds_clean, axis=1)
adv_preds = np.argmax(preds_adv, axis=1)

# Accuracy scores
acc_clean = np.mean(orig_preds == y_test)
acc_adv = np.mean(adv_preds == y_test)

print(f"\n✅ Clean Accuracy: {acc_clean*100:.2f}%")
print(f"⚠️  Adversarial Accuracy: {acc_adv*100:.2f}%")

# Visualization function for multiple examples
def show_examples(orig, adv, label_list, orig_preds, adv_preds, count=6):
    for idx in range(count):
        print(f"\nIndex {idx} | True Label: {label_list[idx]} | Original Pred: {orig_preds[idx]} | Adversarial Pred: {adv_preds[idx]}")

        plt.figure(figsize=(12, 4))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(orig[idx][0], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Adversarial image
        plt.subplot(1, 3, 2)
        plt.imshow(adv[idx][0], cmap='gray')
        plt.title("Adversarial")
        plt.axis('off')

        # Difference heatmap
        diff = adv[idx][0] - orig[idx][0]
        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap='seismic', vmin=-0.3, vmax=0.3)
        plt.title("Difference")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# Show multiple examples
show_examples(x_test, x_test_adv, y_test, orig_preds, adv_preds, count=6)
