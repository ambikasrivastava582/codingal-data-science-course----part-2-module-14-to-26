import pandas as pd
import matplotlib.pyplot as plt


# Read dataset
data = pd.read_csv('Pokemon Data.csv')

# Display first 5 rows
print(data.head())

# Check dataset information
data.info()

# Fill missing values
data['Type 2'] = data['Type 2'].fillna('None')

# Remove Name column
data.drop('Name', axis=1, inplace=True)

# Convert categorical columns into numerical columns
data = pd.get_dummies(data)

# Separate input and output
y = data['Legendary']
X = data.drop('Legendary', axis=1)

# Split data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
ypred = model.predict(X_test)

# Check accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, ypred)

print("Random Forest Accuracy:", accuracy)

# Display actual and predicted values
result = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': ypred
})

print(result.head(10))