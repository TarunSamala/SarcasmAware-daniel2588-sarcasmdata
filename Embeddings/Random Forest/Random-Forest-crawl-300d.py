import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
EMBEDDING_DIM = 300  # crawl-300d-2M dimension
VOCAB_SIZE = 2000000  # Matches the 2M in crawl-300d-2M
OUTPUT_DIR = "sarcasm_randomforest_crawl_outputs"
FASTTEXT_PATH = "../../../crawl-300d-2M.vec"  # Update this path
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
dataset = load_dataset("daniel2588/sarcasmdata")
train = dataset['train']
test = dataset['test']

train_texts = train['text']
train_labels = np.array(train['label'])
test_texts = test['text']
test_labels = np.array(test['label'])

# Load FastText embeddings
def load_fasttext(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

print("Loading FastText embeddings...")
crawl_embeddings = load_fasttext(FASTTEXT_PATH)

# Convert texts to average embedding vectors
def text_to_avg_embedding(texts, embeddings):
    vectors = []
    for text in texts:
        words = text.split()
        word_vectors = [embeddings.get(word, np.zeros(EMBEDDING_DIM)) for word in words]
        avg_vector = np.mean(word_vectors, axis=0)
        vectors.append(avg_vector)
    return np.array(vectors)

# Convert datasets to embedding vectors
print("Converting training texts to embeddings...")
X_train = text_to_avg_embedding(train_texts, crawl_embeddings)
print("Converting test texts to embeddings...")
X_test = text_to_avg_embedding(test_texts, crawl_embeddings)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=150,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# Training
print("Training Random Forest...")
model.fit(X_train, train_labels)

# Save the model and embeddings processor
joblib.dump({
    'model': model,
    'text_processor': text_to_avg_embedding,
    'embeddings_dim': EMBEDDING_DIM
}, os.path.join(OUTPUT_DIR, 'crawl_rf_model.pkl'))

# Generate predictions
print("Generating predictions...")
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