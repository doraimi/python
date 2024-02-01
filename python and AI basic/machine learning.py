from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Preprocess the data by scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a Support Vector Machine (SVM) classifier
svm = SVC(kernel='rbf', gamma='scale', C=1.0)

# Train the classifier
svm.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test_scaled)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""
In this example:

We load the digits dataset, which consists of 8x8 images of handwritten digits along with their corresponding labels (digits 0-9).
We split the dataset into training and testing sets using the train_test_split function.
We preprocess the data by scaling it to have zero mean and unit variance using StandardScaler.
We initialize a Support Vector Machine (SVM) classifier with a radial basis function (RBF) kernel.
We train the SVM classifier on the training data.
We make predictions on the test set.
Finally, we evaluate the accuracy of the classifier by comparing the predicted labels to the true labels using the accuracy_score function.
This example demonstrates a simple machine learning task of classifying images of handwritten digits using a Support Vector Machine classifier. You can experiment with different classifiers, preprocessing techniques, and hyperparameters to improve the accuracy of the model.
"""