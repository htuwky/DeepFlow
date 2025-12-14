import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# ================= 配置参数 =================
CSV_PATH = 'cleaned_traffic_data.csv'
# 这里的 SEQ_LENGTH 要和 LSTM 保持一致，以此构建滞后特征
SEQ_LENGTH = 24
TEST_SIZE = 0.2       # 保持和 LSTM 一样的 20% 测试集

print(">>> 1. Loading and Preparing Data...")
df = pd.read_csv(CSV_PATH)

# 确保按时间排序
if 'date_time' in df.columns:
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')
    # 训练时去掉时间戳字符串，但保留它用于画图
    dates = df['date_time']
    df = df.drop(columns=['date_time'])

# ================= 关键步骤：构建滞后特征 (Lag Features) =================
# LSTM 自动会看过去24小时，XGBoost 需要我们手动把过去24小时的数据“平铺”到当前行
print(f">>> 2. Generating Lag Features (Past {SEQ_LENGTH} hours)...")

for i in range(1, SEQ_LENGTH + 1):
    # shift(i) 表示把数据向下平移 i 行，也就是获取 i 小时前的数据
    df[f'traffic_lag_{i}'] = df['traffic_volume'].shift(i)

# 因为平移，前 24 行会有 NaN (空值)，必须删掉
# 这也符合 LSTM 也是从第 24 行开始预测的逻辑
df = df.dropna()

print(f"Data shape after feature engineering: {df.shape}")

# ================= 准备训练数据 =================
# 目标变量
y = df['traffic_volume']
# 特征变量 (移除目标变量本身，防止作弊)
X = df.drop(columns=['traffic_volume'])

# 严格按照时间切分 (不能 shuffle，因为是时间序列)
# 这与 LSTM 的 X[:train_size] 逻辑完全一致
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ================= 定义与训练 XGBoost =================
print(">>> 3. Training XGBoost...")
start_time = time.time()

# XGBoost 参数配置
model = xgb.XGBRegressor(
    n_estimators=1000,   # 树的数量 (类似 LSTM 的 Epochs)
    learning_rate=0.05,  # 学习率
    max_depth=6,         # 树深 (类似 LSTM 的层数/复杂度)
    subsample=0.8,       # 防止过拟合
    colsample_bytree=0.8,# 防止过拟合
    n_jobs=-1,           # 使用所有 CPU 核心
    early_stopping_rounds=50, # 如果50轮不提升就停止
    random_state=42
)

# 开始训练
# eval_set 用于监控测试集的表现，防止过拟合
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100  # 每100轮打印一次
)

print(f"Training Time: {time.time() - start_time:.2f}s")

# ================= 预测与评估 =================
print(">>> 4. Evaluating...")
preds = model.predict(X_test)

# 计算指标 (和 LSTM 标准一致)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print(f"\n======== XGBoost Test Results ========")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

# ================= 可视化 =================
plt.figure(figsize=(12, 6))
limit = 200
# 还要把 index 重置一下方便画图
y_test_np = y_test.values
preds_np = preds

plt.plot(y_test_np[:limit], label='Actual Traffic', color='blue', alpha=0.7)
plt.plot(preds_np[:limit], label='XGBoost Prediction', color='green', linestyle='--', alpha=0.8)
plt.title('XGBoost Prediction vs Actual (First 200 Hours)')
plt.xlabel('Time (Hours)')
plt.ylabel('Traffic Volume')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig('xgboost_result.png')
print("Result image saved to xgboost_result.png")

# 查看特征重要性 (看看它觉得谁最重要)
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=20, height=0.5)
plt.title("Feature Importance")
plt.savefig('xgboost_importance.png')