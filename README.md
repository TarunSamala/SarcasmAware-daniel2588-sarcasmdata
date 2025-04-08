# Sarcasm Aware Classification Framework

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-blue)](https://scikit-learn.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive sarcasm detection framework implementing multiple deep learning architectures with different embedding schemes.

## Features
**Supported Models**
- Bi-LSTM with Attention Mechanism
- Bi-GRU
- BERT (Base Uncased)
- Random Forest (Traditional ML Baseline)

**Embedding Support**
- GloVe (6B.100D)
- FastText (crawl-300d-2M)
- BERT Tokenizer (Auto-handled)

**Framework Features**
- Cross-architecture evaluation
- Embedding comparison toolkit
- Attention visualization
- GPU/TPU acceleration support
- Comprehensive metrics reporting

## Architectures
| Model | Type | Embedding | Parameters |
|-------|------|-----------|------------|
| Bi-LSTM+Attention | Deep Learning | GloVe/FastText | ~6.5M |
| Bi-GRU | Deep Learning | GloVe/FastText | ~5.8M |
| BERT | Transformer | BERT Tokenizer | ~110M |
| Random Forest | Machine Learning | Averaged Embeddings | 100 Trees |

## Installation
1. Clone repository:
```bash
git clone https://github.com/TarunSamala/SarcasmAware-daniel2588-sarcasmdata.git