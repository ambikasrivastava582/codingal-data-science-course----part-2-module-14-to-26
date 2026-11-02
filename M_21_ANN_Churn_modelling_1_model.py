
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Upload dataset in Google Colab
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('Churn_Modelling.csv')

# 3. Explore dataset
print(df.head())
df.info()
print(df.describe())

# 4. Label Encoding
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df['Geography'] = lb.fit_transform(df['Geography'])
df['Gender'] = lb.fit_transform(df['Gender'])

# 5. Remove unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 6. Separate input (X) and output (y)
y = df.pop('Exited')
X = df

# 7. Split into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 8. Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 9. Build Artificial Neural Network (ANN)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# Hidden layer with 6 neurons
model.add(Dense(6, activation='relu', input_shape=(10,)))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# 10. Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 11. Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 12. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# 13. Predict customer churn
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

print("First 10 predictions:")
print(predictions[:10])
