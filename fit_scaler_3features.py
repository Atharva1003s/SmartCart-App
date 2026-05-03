import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load your data (update the path if needed)
df = pd.read_csv('data/smartcart_customers.csv')

# Feature engineering: create Age if not present
df['Age'] = 2026 - df['Year_Birth']

# Handle missing values for Income (if any)
df['Income'] = df['Income'].fillna(df['Income'].median())

# For demonstration, use 'MntWines' as Spending Score (replace with your actual column if different)
df['Spending_Score'] = df['MntWines']

# Select only the 3 features
X = df[['Age', 'Income', 'Spending_Score']]

# Fit scaler
scaler = StandardScaler()
scaler.fit(X)

# Save scaler
with open('model/scaler_3features.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('Scaler for 3 features saved as model/scaler_3features.pkl')
