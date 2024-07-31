import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from rdkit import Chem

# Define the N-gram length (e.g., 2 for bigrams)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

n_gram_length = 2

# Function to generate N-grams from a SMILES string
def generate_n_grams(smiles, n):
    n_grams = [smiles[i:i+n] for i in range(len(smiles) - n + 1)]
    return n_grams

# Load your dataset (SMILES and labels)
data = pd.read_csv('../../data/NR-AR-LBD_train.smiles', sep='\t', names=['SMILES', 'Label'])

# Convert SMILES strings to N-gram sequences
data['Ngrams'] = data['SMILES'].apply(lambda x: generate_n_grams(x, n_gram_length))

# Create a vocabulary of unique N-grams
vocabulary = set()
for n_grams in data['Ngrams']:
    vocabulary.update(n_grams)

# Create a mapping from N-gram to integer index
n_gram_to_index = {n_gram: i for i, n_gram in enumerate(vocabulary)}

# Encode SMILES strings using the vocabulary
data['Encoded'] = data['Ngrams'].apply(lambda x: [n_gram_to_index[n_gram] for n_gram in x])

# Pad sequences to the same length
max_sequence_length = max(len(seq) for seq in data['Encoded'])
data['Padded'] = data['Encoded'].apply(lambda x: x + [0] * (max_sequence_length - len(x)))

# Convert the padded data to a format suitable for machine learning
X = np.array(data['Padded'].tolist())
y = data['Label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train a Random Forest Classifier
#clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize and train a Support Vector Classification (SVC) model
#clf = SVC(kernel='rbf', random_state=42)

# Initialize and train a Decision Tree Classifier model
#clf = DecisionTreeClassifier(random_state=42)

# Initialize and train a GradientBoostingClassifier model
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classification model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
