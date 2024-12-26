import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# === 데이터셋 준비 === #
class TimeSeriesDataset(Dataset):
    def __init__(self, data, features, target, seq_length=10):
        self.features = data[features].values
        self.targets = data[target].values
        self.seq_length = seq_length

    def __len__(self):
        # 시계열 길이만큼 슬라이딩 윈도우 생성
        return len(self.features) - self.seq_length + 1

    def __getitem__(self, idx):
        # 시계열 데이터 생성
        X = self.features[idx:idx + self.seq_length]  # [seq_length, input_dim]
        y = self.targets[idx + self.seq_length - 1]  # 마지막 시점의 레이블
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# === Transformer 기반 분류기 === #
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input shape: [batch_size, seq_length, input_dim]
        x = self.embedding(x)  # [batch_size, seq_length, d_model]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_length, batch_size, d_model]
        x = self.transformer(x)  # [seq_length, batch_size, d_model]
        x = x.mean(dim=0)  # Global average pooling over the sequence
        return self.fc(x)



# === 데이터 읽기 === #
file_path = "walkvsrun_sorted.csv"
data = pd.read_csv(file_path)


# === 데이터 전처리 === #
# datetime 형식 변환 (date와 time 컬럼이 있다고 가정)
data['full_datetime'] = pd.to_datetime(data['full_datetime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
#data = data.dropna(subset=['datetime'])

# Features와 Target 설정
features = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
target = 'activity'

# 데이터 분할
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])

# Dataset과 DataLoader 생성
train_dataset = TimeSeriesDataset(train_data, features, target)
test_dataset = TimeSeriesDataset(test_data, features, target)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === 모델 학습 === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = len(features)
num_classes = data[target].nunique()
model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

# === 모델 평가 === #
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        _, predicted = torch.max(output, 1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 성능 측정
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')


print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
