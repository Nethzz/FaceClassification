import os
import urllib.request
import tarfile
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Download and extract the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/faces.tar.gz"
dataset_path = "faces.tar.gz"
data_dir = "faces"

if not os.path.exists(data_dir):
    urllib.request.urlretrieve(url, dataset_path)
    with tarfile.open(dataset_path) as tar:
        tar.extractall()


# Function to recursively find JPG files in directories
def find_jpg_files(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))
    return jpg_files


# Find JPG image files
image_files = find_jpg_files(data_dir)
print(f"JPG image files: {image_files}")

# Load the images and preprocess
image_list = []
labels = []

for image_file in image_files:
    # Load image
    image = imread(image_file, as_gray=True)
    # Resize image
    image_resized = resize(image, (32, 32), anti_aliasing=True).flatten()
    image_list.append(image_resized)

    # Extract labels from filename
    filename = os.path.basename(image_file)
    parts = filename.split('.')[0].split('_')
    head_position = parts[1]
    expression = parts[2]
    eye_state = parts[3]
    labels.append([head_position, expression, eye_state])

# Convert to numpy arrays
images = np.array(image_list)
labels = np.array(labels)

# Debug prints
print(f"Images array shape: {images.shape}")
print(f"Labels array shape: {labels.shape}")

# One-hot encoding the labels
head_positions = ['straight', 'left', 'right', 'up']
expressions = ['neutral', 'happy', 'sad', 'angry']
eye_states = ['sunglasses', 'open']

# Create dictionaries to map labels to integer values
head_positions_dict = {label: i for i, label in enumerate(head_positions)}
expressions_dict = {label: i + len(head_positions) for i, label in enumerate(expressions)}
eye_states_dict = {label: i + len(head_positions) + len(expressions) for i, label in enumerate(eye_states)}

# Convert labels to integer values
labels_int = np.zeros((labels.shape[0],), dtype=int)
for i, label in enumerate(labels):
    head_pos_int = head_positions_dict[label[0]]
    expression_int = expressions_dict[label[1]]
    eye_state_int = eye_states_dict[label[2]]
    labels_int[i] = head_pos_int * len(expressions) * len(eye_states) + expression_int * len(eye_states) + eye_state_int

# Debug prints
print("\nLabels as integers:")
print("Shape:", labels_int.shape)
print("Unique labels:", np.unique(labels_int))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_int, test_size=0.2, random_state=42)

print("\nAfter split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Train an SVM
svm_clf = SVC(kernel='rbf')
svm_clf.fit(X_train, y_train)

# Train a Multilayer Perceptron
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
mlp_clf.fit(X_train, y_train)

# Train a Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Evaluate the models
svm_pred = svm_clf.predict(X_test)
mlp_pred = mlp_clf.predict(X_test)
dt_pred = dt_clf.predict(X_test)

print("\nEvaluation Results:")
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("MLP Accuracy:", accuracy_score(y_test, mlp_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# Save the models
os.makedirs("models", exist_ok=True)
joblib.dump(svm_clf, 'models/svm_model.pkl')
joblib.dump(mlp_clf, 'models/mlp_model.pkl')
joblib.dump(dt_clf, 'models/dt_model.pkl')

print("\nModels saved successfully.")
