import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout, SpatialDropout1D  # Changed LSTM to GRU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MAX_LEN = 100
VOCAB_SIZE = 20000
EMBEDDING_DIM = 300
BATCH_SIZE = 256
EPOCHS = 30
OUTPUT_DIR = "sarcasm_fasttext_bigru_outputs"  # Changed output directory
FASTTEXT_PATH = "../../../crawl-300d-2M.vec" # Update this path

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
dataset = load_dataset("daniel2588/sarcasmdata")
train = dataset['train']
test = dataset['test']

train_texts = train['text']
train_labels = np.array(train['label'])
test_texts = test['text']
test_labels = np.array(test['label'])

# Tokenization and sequencing
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_LEN, padding='post')

# Load FastText embeddings
def load_fasttext(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

embeddings = load_fasttext(FASTTEXT_PATH)

# Create embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE and word in embeddings:
        embedding_matrix[i] = embeddings[word]

# FastText-enhanced Bi-LSTM Model
def build_fasttext_bigru_model():  # Renamed function
    inputs = Input(shape=(MAX_LEN,))
    
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                embeddings_initializer=Constant(embedding_matrix),
                trainable=False,
                mask_zero=True)(inputs)
    
    x = SpatialDropout1D(0.3)(x)
    
    # Changed to Bidirectional GRU
    x = Bidirectional(GRU(128,  # Changed layer type
                       kernel_regularizer=regularizers.l2(0.001),
                       recurrent_regularizer=regularizers.l2(0.001),
                       dropout=0.2))(x)
    
    x = Dense(128, activation='relu', 
             kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.h5'), save_best_only=True, monitor='val_accuracy')
]

# Training
model = build_fasttext_bigru_model()
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save training curves
def save_training_curves(history):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Accuracy Curves', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Loss Curves', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

save_training_curves(history)

# Generate and save reports
y_pred = (model.predict(X_test) > 0.5).astype(int)

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
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")