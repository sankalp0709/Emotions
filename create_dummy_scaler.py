import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create a dummy scaler fitted on random data with 10 features
# This is a workaround because the original scaler.pkl is missing
scaler = StandardScaler()
# Fit on some dummy data. Ideally this should be the training data.
# Since we don't have it, we use a range of values to avoid division by zero if variance is 0
dummy_data = np.random.rand(100, 10) 
scaler.fit(dummy_data)

joblib.dump(scaler, 'scaler.pkl')
print("Created dummy scaler.pkl")
