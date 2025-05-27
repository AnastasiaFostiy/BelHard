"""
Используйте данные о зарплате работников и создайте модель регрессии
для прогнозирования зарплаты на основе опыта работы и образования.
"""


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd


path_to_dataset = '/Users/anastasiyafostiy/.cache/kagglehub/datasets/mohithsairamreddy/' \
                  'salary-data/versions/4/Salary_Data.csv'

data = pd.read_csv(path_to_dataset)
data = pd.get_dummies(data, columns=['Education Level'])
data["School"] = data["Education Level_High School"]
data["Bachelor"] = data["Education Level_Bachelor's"] | data["Education Level_Bachelor's Degree"]
data["Master"] = data["Education Level_Master's"] | data["Education Level_Master's Degree"]
data["PhD"] = data["Education Level_PhD"] | data["Education Level_phD"]
data.drop(columns=[
    "Education Level_Bachelor's", "Education Level_Bachelor's Degree", "Education Level_Master's",
    "Education Level_Master's Degree", "Education Level_PhD", "Education Level_phD", "Education Level_High School",
    "Age", "Gender", "Job Title"
], inplace=True)
data.dropna(inplace=True)

X = data.drop(columns=['Salary'])
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}')    # MAE: 21916.73744745448
print(f'MSE: {mse}')    # MSE: 804542957.8631672
print(f'R2: {r2}')      # R2: 0.717005015194855

delta = y_test - y_pred

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(y_test, y_pred, alpha=0.7, color="blue")
axs[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed", label="желаемое значение")
axs[0].set_xlabel("реальные значения MPG")
axs[0].set_ylabel("предсказанные MPG")
axs[0].set_title("реальные vs предсказанные MPG")
axs[0].legend()

axs[1].scatter(y_test, delta, alpha=0.7, color="purple")
axs[1].axhline(y=0, color="red", linestyle="dashed", label="желаемое значение")
axs[1].set_xlabel("реальные значения")
axs[1].set_ylabel("ошибка (y_test - y_pred)")
axs[1].set_title("график ошибок предсказания")
axs[1].legend()

plt.tight_layout()
plt.show()


