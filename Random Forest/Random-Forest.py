import os
import numpy as np
import joblib  # For saving the model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MAX_LEN = 100  # Not used but kept for consistency
VOCAB_SIZE = 20000
OUTPUT_DIR = "sarcasm_randomforest_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
dataset = load_dataset("daniel2588/sarcasmdata")
train = dataset['train']
test = dataset['test']

train_texts = train['text']
train_labels = np.array(train['label'])
test_texts = test['text']
test_labels = np.array(test['label'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=VOCAB_SIZE,
    ngram_range=(1, 2),)  # Using unigrams and bigrams

X_train = tfidf.fit_transform(train_texts)
X_test = tfidf.transform(test_texts)

# Save the vectorizer
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl'))

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=150,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# Training
model.fit(X_train, train_labels)

# Save the model
joblib.dump(model, os.path.join(OUTPUT_DIR, 'random_forest_model.pkl'))

# Generate predictions
y_pred = model.predict(X_test)

# Generate and save reports
# Classification Report
report = classification_report(test_labels, y_pred, target_names=['Not Sarcastic', 'Sarcastic'])
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Sarcastic', 'Sarcastic'],
            yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Final evaluation
test_acc = model.score(X_test, test_labels)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")