import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Cargar los datos
data = pd.read_csv("C:/Users/pepej/Desktop/ExamenMineria/stroke_data.csv")
# en la línea 16 tuve que poner la ruta especifica de la carpeta para que corriera el programa, porque si
# solo ponia stroke_data.csv no lo reconocía


imputer = SimpleImputer(strategy='mean')
data['bmi'] = imputer.fit_transform(data[['bmi']])

scaler = MinMaxScaler()
numeric_columns = ['age', 'avg_glucose_level', 'bmi']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(ax=axes[0], data=data, y='age', color='lightblue')
axes[0].set_title('Boxplot de Edad (age)')

sns.boxplot(ax=axes[1], data=data, y='avg_glucose_level', color='lightgreen')
axes[1].set_title('Boxplot de Nivel Promedio de Glucosa (avg_glucose_level)')

sns.boxplot(ax=axes[2], data=data, y='bmi', color='lightcoral')
axes[2].set_title('Boxplot de Índice de Masa Corporal (bmi)')

plt.tight_layout()
plt.show()

def eliminar_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

data_clean = data.copy()
for column in numeric_columns:
    data_clean = eliminar_outliers(data_clean, column)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.boxplot(ax=axes[0, 0], data=data, y='age', color='lightblue')
axes[0, 0].set_title('Antes - Edad')

sns.boxplot(ax=axes[0, 1], data=data, y='avg_glucose_level', color='lightgreen')
axes[0, 1].set_title('Antes - Glucosa')

sns.boxplot(ax=axes[0, 2], data=data, y='bmi', color='lightcoral')
axes[0, 2].set_title('Antes - BMI')

# Gráficos después
sns.boxplot(ax=axes[1, 0], data=data_clean, y='age', color='lightblue')
axes[1, 0].set_title('Después - Edad')

sns.boxplot(ax=axes[1, 1], data=data_clean, y='avg_glucose_level', color='lightgreen')
axes[1, 1].set_title('Después - Glucosa')

sns.boxplot(ax=axes[1, 2], data=data_clean, y='bmi', color='lightcoral')
axes[1, 2].set_title('Después - BMI')

plt.tight_layout()
plt.show()

data_clean = pd.get_dummies(data_clean, columns=['gender', 'ever_married', 'Residence_type'], drop_first=True)

correlation_matrix = data_clean.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap de Correlaciones')
plt.show()

X = data_clean[['avg_glucose_level']]
y = data_clean['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Intercepto:", lr.intercept_)
print("Coeficiente:", lr.coef_)
print(f"R^2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

X_multiple_1 = data_clean[['age', 'avg_glucose_level', 'bmi']]
X_multiple_2 = data_clean[['age', 'avg_glucose_level', 'bmi', 'gender_Male']]
X_multiple_3 = data_clean[['age', 'avg_glucose_level', 'bmi', 'gender_Male', 'ever_married_Yes']]

def multiple_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {'R^2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

results_1 = multiple_regression(X_multiple_1, y)
results_2 = multiple_regression(X_multiple_2, y)
results_3 = multiple_regression(X_multiple_3, y)

comparison = pd.DataFrame({
    'Modelo 1': results_1,
    'Modelo 2': results_2,
    'Modelo 3': results_3
})

print(comparison)





