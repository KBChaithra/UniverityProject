import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, recall_score, precision_score
from gensim.models import Word2Vec
from rdkit import Chem

# Load your dataset (SMILES and labels)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../../data/NR-AR-LBD_train.smiles', sep='\t', names=['SMILES', 'Label'])

# Tokenize SMILES strings into characters
data['Tokens'] = data['SMILES'].apply(list)

# Train a Word2Vec model on the character sequences
model = Word2Vec(data['Tokens'], vector_size=100, window=5, min_count=1, sg=0)

# Function to convert a SMILES string to a Word2Vec embedding
def smiles_to_word2vec_embedding(smiles, model):
    tokens = list(smiles)
    embedding = np.zeros(model.vector_size)
    count = 0
    for token in tokens:
        if token in model.wv:
            embedding += model.wv[token]
            count += 1
    if count > 0:
        embedding /= count
    return embedding

# Encode SMILES strings using Word2Vec embeddings
data['Encoded'] = data['SMILES'].apply(lambda x: smiles_to_word2vec_embedding(x, model))

# Convert the encoded data to a format suitable for machine learning
X = np.array(data['Encoded'].tolist())
y = data['Label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier (or another classifier)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize and train a Support Vector Classification (SVC) model
#clf = SVC(kernel='rbf', random_state=42)

# Initialize and train a Decision Tree Classifier model
#clf = DecisionTreeClassifier(random_state=42)

# Initialize and train a GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classification model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print(f'F1-Score: {f1:.4f}')

# Calculate Recall
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.4f}')

# Calculate Precision
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.4f}')

# Calculate Mean Squared Error (MSE) - if needed
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")






