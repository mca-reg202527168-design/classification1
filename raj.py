import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Generate some random 'features' (e.g., test scores)
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # 1 if sum > 1, else 0

# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Initialize and Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Make Predictions
predictions = model.predict(X_test)

# 5. Output Results
accuracy = accuracy_score(y_test, predictions)
print(f"--- Classification Results ---")
print(f"Model Accuracy: {accuracy * 100}%")
print(f"Predictions: {predictions}")