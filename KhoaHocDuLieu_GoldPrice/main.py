import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# Thiết lập phong cách đồ thị
sns.set_theme(style="whitegrid")

# TỰ ĐỘNG TẠO THƯ MỤC LƯU ẢNH (Nếu chưa có)
output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"-> Đã tạo thư mục '{output_dir}' thành công.")

print("--- ĐANG ĐỌC VÀ XỬ LÝ DỮ LIỆU ---")
# Đọc dữ liệu
df = pd.read_csv("gold_prices_1995_2026_feb.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2000-01-01'].copy()
df.rename(columns={'Gold_Price_USD_YFinance': 'Close'}, inplace=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

print("\n" + "="*50)
print("PHẦN 1: PHÂN TÍCH DỮ LIỆU EDA (5 CÂU HỎI)")
print("="*50)

# ---------------------------------------------------------
# CÂU 1: XU HƯỚNG DÀI HẠN
# ---------------------------------------------------------
print("1. Đang xử lý câu 1 -> Xu hướng dài hạn...")
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'], color='goldenrod', linewidth=2)
plt.title("Câu 1: Xu hướng giá vàng thế giới (2000 - 2026)", fontsize=14, fontweight='bold')
plt.xlabel("Năm")
plt.ylabel("Giá (USD)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "q1_trend.png"), dpi=300)
plt.close() # Đóng luồng vẽ để giải phóng bộ nhớ

# ---------------------------------------------------------
# CÂU 2: NĂM CÓ GIÁ TRUNG BÌNH CAO NHẤT
# ---------------------------------------------------------
print("2. Đang xử lý câu 2 -> Giá trung bình theo năm...")
yearly_mean = df.groupby('Year')['Close'].mean()
max_year = yearly_mean.idxmax()

plt.figure(figsize=(12, 5))
colors = ['red' if year == max_year else 'skyblue' for year in yearly_mean.index]
sns.barplot(x=yearly_mean.index, y=yearly_mean.values, palette=colors)
plt.title(f"Câu 2: Giá vàng trung bình theo năm (Cao nhất: {max_year})", fontsize=14, fontweight='bold')
plt.xlabel("Năm")
plt.ylabel("Giá trung bình (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "q2_top10.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# CÂU 3: NĂM CÓ BIẾN ĐỘNG MẠNH NHẤT
# ---------------------------------------------------------
print("3. Đang xử lý câu 3 -> Mức độ biến động rủi ro...")
yearly_std = df.groupby('Year')['Close'].std().fillna(0)
max_std_year = yearly_std.idxmax()

plt.figure(figsize=(12, 5))
colors_std = ['orange' if year == max_std_year else 'lightgray' for year in yearly_std.index]
sns.barplot(x=yearly_std.index, y=yearly_std.values, palette=colors_std)
plt.title(f"Câu 3: Độ biến động giá vàng theo năm (Mạnh nhất: {max_std_year})", fontsize=14, fontweight='bold')
plt.xlabel("Năm")
plt.ylabel("Độ lệch chuẩn (Mức độ biến động)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "q3_death_rate.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# CÂU 4: THÁNG THƯỜNG CÓ GIÁ CAO NHẤT
# ---------------------------------------------------------
print("4. Đang xử lý câu 4 -> Tính mùa vụ theo tháng...")
monthly_mean = df.groupby('Month')['Close'].mean()
max_month = monthly_mean.idxmax()

plt.figure(figsize=(10, 5))
colors_month = ['purple' if month == max_month else 'lightblue' for month in monthly_mean.index]
sns.barplot(x=monthly_mean.index, y=monthly_mean.values, palette=colors_month)
plt.title(f"Câu 4: Tính mùa vụ - Giá vàng trung bình theo tháng", fontsize=14, fontweight='bold')
plt.xlabel("Tháng")
plt.ylabel("Giá trung bình (USD)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "q4_deaths.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# CÂU 5: CÁC SỰ KIỆN LỚN TÁC ĐỘNG
# ---------------------------------------------------------
print("5. Đang xử lý câu 5 -> Các sự kiện tác động...")
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], color='gray', linewidth=1.5, label='Giá vàng')
plt.axvline(pd.to_datetime('2008-09-15'), color='red', linestyle='--', linewidth=2, label='Khủng hoảng 2008')
plt.axvline(pd.to_datetime('2020-03-11'), color='blue', linestyle='--', linewidth=2, label='COVID-19 (2020)')
plt.axvline(pd.to_datetime('2024-01-01'), color='purple', linestyle='--', linewidth=2, label='Lạm phát & Bất ổn (2024)')
plt.title("Câu 5: Giá vàng và các sự kiện lịch sử", fontsize=14, fontweight='bold')
plt.xlabel("Năm")
plt.ylabel("Giá (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "q5_injuries.png"), dpi=300)
plt.close()


print("\n" + "="*50)
print("PHẦN 2: MACHINE LEARNING (2 CÂU HỎI)")
print("="*50)

# ---------------------------------------------------------
# CÂU 6: DỰ ĐOÁN GIÁ (LR, ARIMA, RF)
# ---------------------------------------------------------
print("6. Đang xử lý câu 6 -> Huấn luyện các mô hình dự đoán...")
df_ml = df.dropna().copy()
split_idx = int(len(df_ml) * 0.8)
train_data, test_data = df_ml.iloc[:split_idx], df_ml.iloc[split_idx:]

X_train = np.arange(len(train_data)).reshape(-1, 1)
y_train = train_data['Close'].values
X_test = np.arange(len(train_data), len(df_ml)).reshape(-1, 1)
y_test = test_data['Close'].values

# 1. Linear Regression
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"   - Sai số Linear Regression (MAE): {mean_absolute_error(y_test, lr_pred):.2f} USD")

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"   - Sai số Random Forest (MAE): {mean_absolute_error(y_test, rf_pred):.2f} USD")

# 3. ARIMA
arima_model = ARIMA(y_train, order=(1, 1, 1)).fit()
arima_pred = arima_model.forecast(steps=len(test_data))
print(f"   - Sai số ARIMA (MAE): {mean_absolute_error(y_test, arima_pred):.2f} USD")

plt.figure(figsize=(12, 6))
plt.plot(test_data['Date'], y_test, color='black', label='Thực tế', linewidth=2)
plt.plot(test_data['Date'], lr_pred, color='blue', linestyle='--', label='Linear Regression')
plt.plot(test_data['Date'], rf_pred, color='green', linestyle='-.', label='Random Forest')
plt.plot(test_data['Date'], arima_pred, color='red', linestyle=':', label='ARIMA')
plt.title("Câu 6: So sánh kết quả các mô hình dự đoán (Tập Test)", fontsize=14, fontweight='bold')
plt.xlabel("Thời gian")
plt.ylabel("Giá (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "q6_predict_2026.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# CÂU 7: PHÂN CỤM K-MEANS
# ---------------------------------------------------------
print("7. Đang xử lý câu 7 -> Phân cụm cấu trúc K-Means...")
annual_stats = pd.DataFrame({'MeanPrice': yearly_mean, 'SdPrice': yearly_std}).dropna()
scaled_data = StandardScaler().fit_transform(annual_stats)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
annual_stats['Cluster'] = kmeans.fit_predict(scaled_data)

cluster_mapping = annual_stats.groupby('Cluster')['SdPrice'].median().sort_values()
labels = ["Biến động thấp", "Biến động trung bình", "Biến động cao"]
label_dict = {cluster_id: label for cluster_id, label in zip(cluster_mapping.index, labels)}
annual_stats['Label'] = annual_stats['Cluster'].map(label_dict)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=annual_stats, x='MeanPrice', y='SdPrice', hue='Label', s=150, palette=['green', 'orange', 'red'])
for i in range(len(annual_stats)):
    plt.text(annual_stats['MeanPrice'].iloc[i], annual_stats['SdPrice'].iloc[i] + 5, str(annual_stats.index[i]), fontsize=9)
plt.title('Câu 7: Phân cụm các năm theo K-Means (Dựa vào Giá và Biến động)', fontsize=14, fontweight='bold')
plt.xlabel('Giá vàng trung bình (USD)')
plt.ylabel('Độ lệch chuẩn (Biến động)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "q7_clustering.png"), dpi=300)
plt.close()

print("\n" + "="*50)
print("HOÀN THÀNH TOÀN BỘ PHÂN TÍCH VÀ ĐÃ LƯU 7 ẢNH VÀO THƯ MỤC 'images'!")
print("="*50)