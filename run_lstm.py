import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time

# ================= 配置参数 =================
CSV_PATH = 'cleaned_traffic_data.csv'
SEQ_LENGTH = 24  # 用过去24小时
PREDICT_DIST = 1  # 预测未来1小时
BATCH_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 200  # 训练20轮 (演示用，你可以改大)
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ================= 1. 数据准备 =================
print(">>> 1. Loading Data...")
df = pd.read_csv(CSV_PATH)

# 去掉 date_time 列，保留所有数值型特征
feature_cols = [c for c in df.columns if c != 'date_time']
target_col = 'traffic_volume'

# 确保数据是 float 类型
data = df[feature_cols].values.astype(float)
target_idx = feature_cols.index(target_col)

# 归一化 (Scaler)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# 制作时间序列数据集
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - PREDICT_DIST + 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length][target_idx]  # 只预测流量这一列
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


X, y = create_sequences(data_scaled, SEQ_LENGTH)

# 转为 Tensor
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).view(-1, 1)

# 划分训练/测试集 (前80%训练)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# ================= 2. 模型定义 =================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out


input_dim = X_train.shape[2]
model = LSTMModel(input_dim, HIDDEN_SIZE, NUM_LAYERS, 1).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================= 3. 训练循环 =================
print(">>> 2. Start Training...")
loss_history = []
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    epoch_loss = np.mean(batch_losses)
    loss_history.append(epoch_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

print(f"Training Time: {time.time() - start_time:.1f}s")

# ================= 4. 预测与评估 =================
print(">>> 3. Evaluating...")
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        out = model(x_batch).cpu().numpy()
        predictions.extend(out)
        actuals.extend(y_batch.numpy())

predictions = np.array(predictions)
actuals = np.array(actuals)


# --- 关键步骤：反归一化 (Inverse Scaling) ---
# 我们只对 traffic_volume 进行了预测，但 scaler 是对所有列 fit 的
# 所以我们需要构建一个 dummy 矩阵来反变换
def inverse_transform_target(pred_array, scaler, target_col_idx, n_features):
    # 创建一个形状为 (N, features) 的全零矩阵
    dummy = np.zeros((len(pred_array), n_features))
    # 把预测值填入 traffic_volume 所在的列
    dummy[:, target_col_idx] = pred_array.flatten()
    # 反变换
    original = scaler.inverse_transform(dummy)
    # 取出 traffic_volume 列
    return original[:, target_col_idx]


# 还原数值
real_preds = inverse_transform_target(predictions, scaler, target_idx, input_dim)
real_actuals = inverse_transform_target(actuals, scaler, target_idx, input_dim)

# 计算指标
rmse = np.sqrt(mean_squared_error(real_actuals, real_preds))
mae = mean_absolute_error(real_actuals, real_preds)
print(f"\n======== LSTM Test Results ========")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

# ================= 5. 可视化 =================
plt.figure(figsize=(12, 6))
# 只画前 200 个小时，不然太密看不清
limit = 200
plt.plot(real_actuals[:limit], label='Actual Traffic', color='blue', alpha=0.7)
plt.plot(real_preds[:limit], label='LSTM Prediction', color='red', linestyle='--', alpha=0.8)
plt.title('LSTM Prediction vs Actual (First 200 Hours)')
plt.xlabel('Time (Hours)')
plt.ylabel('Traffic Volume')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()  # 如果是命令行运行，这一步会弹窗
plt.savefig('lstm_result.png')  # 保存图片
print("Result image saved to lstm_result.png")