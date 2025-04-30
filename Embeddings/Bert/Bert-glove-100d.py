import os
import numpy as np
import tensorflow as tf
import re
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
MAX_LEN = 35
BATCH_SIZE = 32
EPOCHS = 10
OUTPUT_DIR = "sarcasm_outputs_distilbert"
MODEL_NAME = "distilbert-base-uncased"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data
dataset = load_dataset("daniel2588/sarcasmdata")
train = dataset['train']
test = dataset['test']

train_texts = [clean_text(t) for t in train['text']]
test_texts = [clean_text(t) for t in test['text']]
train_labels = np.array(train['label'])
test_labels = np.array(test['label'])

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def encode_texts(texts):
    return tokenizer(
        texts,
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

# Encode datasets
train_encodings = encode_texts(train_texts)
test_encodings = encode_texts(test_texts)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask']
    },
    train_labels
)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask']
    },
    test_labels
)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Enhanced DistilBERT model
def build_distilbert_model():
    model = TFDistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        dropout=0.2,
        attention_dropout=0.2
    )
    
    # Custom classifier with regularization
    model.classifier = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    return model

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3,
                 min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=2, min_lr=1e-6)
]

# Training
model = build_distilbert_model()
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save training curves
plt.figure(figsize=(15, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves', pad=10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.5, 1.0)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves', pad=10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1.5)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves_distilbert.png'), dpi=300, bbox_inches='tight')
plt.close()

# Generate predictions
raw_preds = model.predict(test_dataset)
y_pred = (raw_preds.logits > 0.5).astype(int)

# Classification Report
report = classification_report(test_labels, y_pred, 
                               target_names=['Not Sarcastic', 'Sarcastic'])
report_path = os.path.join(OUTPUT_DIR, 'classification_report_distilbert.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report (DistilBERT):\n")
    f.write(report)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Not Sarcastic', 'Sarcastic'],
           yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.title('Confusion Matrix (DistilBERT)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_distilbert.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll DistilBERT outputs saved to: {os.path.abspath(OUTPUT_DIR)}")