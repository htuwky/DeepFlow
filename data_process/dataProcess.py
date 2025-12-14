import pandas as pd
import numpy as np


def clean_and_process_data(input_file, output_file):
    print(">>> 1. 正在加载数据...")
    df = pd.read_csv(input_file)
    print(f"原始数据大小: {df.shape}")

    # ---------------------------------------------------------
    # 第一步：时间序列处理与去重
    # ---------------------------------------------------------
    print(">>> 2. 处理时间戳与去重...")
    # 转换 date_time 列为 datetime 对象
    df['date_time'] = pd.to_datetime(df['date_time'])

    # 这一步非常重要：按时间排序，防止数据乱序
    df = df.sort_values('date_time')

    # 去除重复的时间戳
    # 逻辑：同一小时内可能有多次天气记录，但流量是一样的，保留第一条即可
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['date_time'], keep='first')
    print(f"已移除重复行数: {initial_rows - len(df)}")

    # ---------------------------------------------------------
    # 第二步：异常值处理
    # ---------------------------------------------------------
    print(">>> 3. 处理异常值 (0K 温度)...")
    # 统计有多少 0K 温度
    zero_k_count = (df['temp'] == 0).sum()
    if zero_k_count > 0:
        # 将 0 替换为 NaN，然后使用线性插值 (interpolate) 用前后的温度填充
        df['temp'] = df['temp'].replace(0, np.nan)
        df['temp'] = df['temp'].interpolate(method='linear')
        print(f"已修复 {zero_k_count} 个温度异常值 (0 Kelvin)")

    # ---------------------------------------------------------
    # 第三步：特征工程 (Feature Engineering)
    # ---------------------------------------------------------
    print(">>> 4. 提取时间特征与编码...")

    # 3.1 提取时间特征 (这对 XGBoost 和 LSTM 都至关重要)
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek  # 0=周一, 6=周日
    df['month'] = df['date_time'].dt.month

    # 3.2 处理节假日 (Holiday)
    # 原始数据里非节日是 "None"
    # 策略：生成二分类特征 is_holiday (1=是节日, 0=不是)
    df['is_holiday'] = df['holiday'].apply(lambda x: 0 if x == 'None' else 1)

    # 3.3 处理天气 (Weather) - One-Hot Encoding
    # 将分类变量 weather_main (如 Rain, Clear) 转换为数值列
    weather_dummies = pd.get_dummies(df['weather_main'], prefix='weather')
    # 将生成的哑变量拼接到主表
    df = pd.concat([df, weather_dummies], axis=1)

    # ---------------------------------------------------------
    # 第四步：整理与保存
    # ---------------------------------------------------------
    print(">>> 5. 整理最终列并保存...")

    # 定义我们最终需要的特征列
    # 注意：我们保留 date_time 是为了后续可视化或检查，但在训练模型时通常要去掉它
    features_to_keep = [
                           'date_time',  # 时间索引
                           'traffic_volume',  # 【目标变量】
                           'temp',  # 气温
                           'rain_1h',  # 降雨量
                           'snow_1h',  # 降雪量
                           'clouds_all',  # 云量
                           'hour',  # 小时 (关键)
                           'day_of_week',  # 星期几
                           'month',  # 月份
                           'is_holiday'  # 是否节假日
                       ] + list(weather_dummies.columns)  # 加上所有的天气列

    final_df = df[features_to_keep]

    # 保存清洗后的数据
    final_df.to_csv(output_file, index=False)
    print(f"处理完成！文件已保存为: {output_file}")
    print(f"最终数据形状: {final_df.shape}")
    print("-" * 30)
    print("前5行预览:")
    print(final_df.head())


if __name__ == "__main__":
    # 输入文件名 (请确保该文件在当前目录下)
    INPUT_FILE = './Metro_Interstate_Traffic_Volume.csv'
    # 输出文件名
    OUTPUT_FILE = './cleaned_traffic_data.csv'

    try:
        clean_and_process_data(INPUT_FILE, OUTPUT_FILE)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{INPUT_FILE}'。请确保 CSV 文件在当前文件夹中。")