import pickle
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import matplotlib.pyplot as plt
import seaborn as sns


# Load data and labels from the pickle file
data_dict = pickle.load(open('dataset.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train classifier using Grid Search for best parameters

parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001],'kernel': ['rbf']}  # Define the parameters for Grid Search

classifier = SVC(probability=True)  # Define the SVM model
grid_search = GridSearchCV(classifier, parameters, cv=10, n_jobs=-1, refit=True, verbose=2)  # Execute Grid Search

start_time = time.time()  # Start time before training
grid_search.fit(x_train, y_train)
end_time = time.time()  # End time after training

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best parameters: {best_params}')
print(f'Best cross-validation score: {best_score}')

# Train classifier with best parameters
best_svm_classifier = grid_search.best_estimator_
best_svm_classifier.fit(x_train, y_train)

# Save the best classifier
with open('svm_letters_classifier.p', 'wb') as f:
    pickle.dump({'model': best_svm_classifier}, f)

training_time = end_time - start_time  # Calculate the training time
print(f'Training time: {training_time} seconds')

# Predict and evaluate accuracy
y_pred_svm = best_svm_classifier.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm) * 100  # Convert accuracy to percentage

print(f'SVM accuracy: {accuracy_svm:.2f}%')

# Calculate confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred_svm)
class_report = classification_report(y_test, y_pred_svm)

# Save confusion matrix and classification report to a text file
with open('classification_report.txt', 'w') as f:
    f.write(f'Best parameters: {best_params}\n\n')
    f.write(f'Best cross-validation score: {best_score}\n\n')
    f.write(f'Training time: {training_time:.2f} seconds\n\n')
    f.write(f'SVM accuracy: {accuracy_svm:.2f}\n\n')
    f.write(f'Total data: {len(data)}\n')
    f.write(f'Training data: {len(x_train)}\n')
    f.write(f'Testing data: {len(x_test)}\n\n')
    f.write(f'Confusion Matrix:\n{conf_matrix}\n\n')
    f.write(f'Classification Report:\n{class_report}\n')

# Print total number of data, training data, and testing data
total_data = len(data)
train_data = len(x_train)
test_data = len(x_test)

print(f'Total data: {total_data}')
print(f'Training data: {train_data}')
print(f'Testing data: {test_data}')

# Plot confusion matrix
labels = ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك',
          'ل', 'م', 'ن', 'هـ', 'و', 'ي', 'لا', 'ة']
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(labels),
            yticklabels=np.unique(labels))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Arabic Letters - SVM')
plt.savefig('confusion_matrix_svm.png')
plt.show()