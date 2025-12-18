import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv('dataset.csv')


#Renaming of the colummns for easier access
data = data.rename(columns={
        'X2 house age': 'House Age',
        'X3 distance to the nearest MRT station': 'Distance to nearest MRT station',
        'Y house price of unit area': 'Price'
    })

# --- 2. Feature Selection ---
X = data[['House Age', 'Distance to nearest MRT station']]
y = data['Price']

# --- 3. Split Data ---
# 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# --- 4. Normalization (Standard Scaling) ---
# Rubric: fit_transform on training set, then transform on test set
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Model Evaluation ---
# Report Accuracy (R2 and RMSE) for both training and test sets

# This where Predictions happens
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Metrics Calculation
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Evaluation")
print(f"Training R-Squared  : {train_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print("-" * 25)
print(f"Testing R-Squared   : {test_r2:.4f}")
print(f"Testing RMSE : {test_rmse:.4f}")

# --- 7. 3D Visualization ---
# Rubric: 3D plot with axes representing the ORIGINAL scale of the data

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting Scatter Points (Train & Test)
#  We use X_train and X_test (unscaled) here so the axes show real years/meters
ax.scatter(X_train['House Age'], X_train['Distance to nearest MRT station'], y_train, 
        c='blue', marker='o', alpha=0.5, label='Training Data')
ax.scatter(X_test['House Age'], X_test['Distance to nearest MRT station'], y_test, 
        c='red', marker='^', alpha=0.5, label='Test Data')

# Plotting the Prediction Surface
# We create a grid across the range of the ORIGINAL data
x1_range = np.linspace(X['House Age'].min(), X['House Age'].max(), 30)
x2_range = np.linspace(X['Distance to nearest MRT station'].min(), X['Distance to nearest MRT station'].max(), 30)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# Flatten the grid to feed into the model
mesh_data = pd.DataFrame({
        'House Age': x1_mesh.ravel(), 
        'Distance to nearest MRT station': x2_mesh.ravel()
})

# CRITICAL STEP: The model was trained on SCALED data.
# We must scale our meshgrid using the SAME scaler before predicting.
mesh_data_scaled = scaler.transform(mesh_data)

# Predict the price (Z-axis)
z_mesh = model.predict(mesh_data_scaled)
z_mesh = z_mesh.reshape(x1_mesh.shape)

# Plot the surface
ax.plot_surface(x1_mesh, x2_mesh, z_mesh, alpha=0.3, color='orange', edgecolor='none')

# Labels and Title
ax.set_xlabel('House Age (years)')
ax.set_ylabel('Distance to MRT (meters)')
ax.set_zlabel('Price')
ax.set_title('Taiwan House Price Prediction (Multiple Linear Regression)')
plt.legend()
plt.show()