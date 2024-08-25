import pandas as pd
import chardet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Detect the encoding
with open('dataset/SeoulBikeData.csv', 'rb') as file:
    encoding_result = chardet.detect(file.read())
    # print(encoding_result['encoding'])


data = pd.read_csv('dataset/SeoulBikeData.csv', encoding='ISO-8859-1')

# Basic Data Overview
print("First few rows of the dataset:")
print(data.head())
print("\nSummary statistics of numerical columns:")
print(data.describe())
print("\nInformation about the dataset:")
print(data.info())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
print(data.head())

# Convert categorical columns
data['Seasons'] = data['Seasons'].astype('category')
data['Holiday'] = data['Holiday'].astype('category')
data['Functioning Day'] = data['Functioning Day'].astype('category')
print(data.head())

# Extract month, day of the week
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek
print(data.head())

# EDA: Visualizing the Target Variable (Rented Bike Count)
plt.figure(figsize=(10, 6))
sns.histplot(data['Rented Bike Count'], kde=True, color='blue')
plt.title('Distribution of Rented Bike Count')
plt.xlabel('Rented Bike Count')
plt.ylabel('Frequency')
plt.show()

# EDA: Correlation Matrix
# Select only numerical columns for correlation
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# EDA: Relationships Between Variables
# Scatter plot of Temperature vs Rented Bike Count
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Temperature(°C)', y='Rented Bike Count', data=data)
plt.title('Temperature vs Rented Bike Count')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rented Bike Count')
plt.show()

# Scatter plot of Humidity vs Rented Bike Count
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Humidity(%)', y='Rented Bike Count', data=data)
plt.title('Humidity vs Rented Bike Count')
plt.xlabel('Humidity (%)')
plt.ylabel('Rented Bike Count')
plt.show()

# EDA: Categorical Data Exploration
# Bar plot for Seasons vs Rented Bike Count
plt.figure(figsize=(8, 6))
sns.barplot(x='Seasons', y='Rented Bike Count', data=data)
plt.title('Seasons vs Rented Bike Count')
plt.xlabel('Seasons')
plt.ylabel('Average Rented Bike Count')
plt.show()

# Bar plot for Holiday vs Rented Bike Count
plt.figure(figsize=(8, 6))
sns.barplot(x='Holiday', y='Rented Bike Count', data=data)
plt.title('Holiday vs Rented Bike Count')
plt.xlabel('Holiday')
plt.ylabel('Average Rented Bike Count')
plt.show()

# Bar plot for Functioning Day vs Rented Bike Count
plt.figure(figsize=(8, 6))
sns.barplot(x='Functioning Day', y='Rented Bike Count', data=data)
plt.title('Functioning Day vs Rented Bike Count')
plt.xlabel('Functioning Day')
plt.ylabel('Average Rented Bike Count')
plt.show()

# EDA: Time Series Analysis
# Plot Rented Bike Count by hour
plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='Rented Bike Count', data=data)
plt.title('Rented Bike Count by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Rented Bike Count')
plt.show()

# Plot Rented Bike Count by day of the week
plt.figure(figsize=(10, 6))
sns.lineplot(x='DayOfWeek', y='Rented Bike Count', data=data)
plt.title('Rented Bike Count by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Rented Bike Count')
plt.show()

# Plot Rented Bike Count by month
plt.figure(figsize=(10, 6))
sns.lineplot(x='Month', y='Rented Bike Count', data=data)
plt.title('Rented Bike Count by Month')
plt.xlabel('Month')
plt.ylabel('Rented Bike Count')
plt.show()

# EDA: Outlier Detection
# Box plot for Rented Bike Count
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Rented Bike Count'])
plt.title('Box Plot of Rented Bike Count')
plt.xlabel('Rented Bike Count')
plt.show()

# One-hot encoding for categorical variables
data = pd.get_dummies(data,
                      columns=['Seasons', 'Holiday', 'Functioning Day'],
                      drop_first=True)
print("\nDataset after one-hot encoding:")
print(data.head())

# Scale numerical Features
scaler = StandardScaler()
numerical_columns = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
                     'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)',
                     'Rainfall(mm)', 'Snowfall (cm)']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
print("\nDataset after scaling numerical features:")
print(data.head())

X = data.drop(['Rented Bike Count', 'Date'], axis=1)  # Features
y = data['Rented Bike Count']  # Target variable

# Split into Training, Validation and Test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

