# Quora-Duplicate-Question-Detection-with-Semantic-Search

# Quora Duplicate Question Detection with Semantic Search

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance semantic retrieval system for detecting duplicate questions on Quora using fine-tuned Sentence-BERT and FAISS for efficient vector search.

## 🚀 Overview

This project implements an end-to-end semantic search system designed to identify duplicate questions in the Quora Question Pairs dataset. By leveraging a fine-tuned Sentence-BERT model with triplet loss and FAISS for scalable vector retrieval, the system achieves strong performance with an **MRR of 0.6401** and **Recall@1 of 0.5065**.

### Key Features

- **🔥 Fine-tuned Sentence-BERT**: Uses `all-MiniLM-L6-v2` optimized with triplet loss for domain-specific similarity understanding
- **⚡ Efficient Retrieval**: FAISS-powered vector search for sub-second query processing
- **📊 Comprehensive Evaluation**: Full suite of IR metrics (MRR, Recall@k, Precision@k, NDCG@k, MAP@k)
- **🎯 Production-Ready**: Scalable bi-encoder architecture suitable for real-world deployment
- **📈 Strong Performance**: Significantly outperforms TF-IDF baseline across all metrics

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch 1.12, Transformers 4.21.2
- **Sentence Embeddings**: Sentence-Transformers 2.2.2
- **Vector Search**: FAISS-GPU 1.7.2
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Optimization**: AdamW optimizer with 2e-5 learning rate

## 📦 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (for `faiss-gpu`)
- 8GB+ RAM recommended

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/quora-semantic-search.git
   cd quora-semantic-search
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: If you don't have a CUDA-enabled GPU, replace `faiss-gpu==1.7.2` with `faiss-cpu==1.7.2` in `requirements.txt` before installation.

## 🚀 Quick Start

### 1. Data Preparation
```bash
python QuoraDataPreparator.py
```
This script will:
- Download and process the Quora Question Pairs dataset
- Create training triplets (72,810 triplets generated)
- Split data into training, knowledge base, and test sets

### 2. Model Training
```bash
python BertModel.py
```
This will:
- Fine-tune the Sentence-BERT model using triplet loss
- Train for 1 epoch with average loss of 3.7671
- Save the fine-tuned model to `./models/finetuned-all-MiniLM-L6-v2`

### 3. Evaluation and Search
```bash
python QA.py
```
This will:
- Build FAISS index with 297,750 question vectors
- Evaluate on 44,663 test queries
- Display comprehensive performance metrics

## 📁 Project Structure

```
quora-semantic-search/
├── QuoraDataPreparator.py    # Data processing and triplet generation
├── BertModel.py              # Model training and fine-tuning
├── QA.py                     # Evaluation and question answering
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── models/                   # Saved model checkpoints
│   └── finetuned-all-MiniLM-L6-v2/
├── data/                     # Dataset files (auto-generated)
│   ├── train_triplets.csv
│   ├── knowledge_base.csv
│   └── test_queries.csv
└── results/                  # Evaluation outputs
```

## 📊 Performance Results

| Metric | TF-IDF Baseline | Our Model | Improvement |
|--------|----------------|-----------|------------|
| **MRR** | 0.4482 | **0.6401** | +42.8% |
| **Recall@1** | 0.3156 | **0.5065** | +60.5% |
| **Recall@5** | 0.6319 | **0.8226** | +30.2% |
| **Recall@10** | 0.7428 | **0.9115** | +22.7% |

### Key Achievements
- **50.65%** of queries return a correct duplicate as the top result
- **82.26%** of queries find a correct duplicate within top-5 results
- **91.15%** of queries find a correct duplicate within top-10 results

## 🔬 System Architecture

The system follows a bi-encoder approach:

1. **Offline Phase**: All knowledge base questions are encoded into 384-dimensional vectors and indexed using FAISS
2. **Online Phase**: Query questions are encoded using the same model, and FAISS performs fast similarity search

## 💡 Usage Examples

### Basic Question Search
```python
from BertModel import BertModel

# Initialize the system
qa_system = BertModel()
qa_system.load_fine_tuned_model()

# Search for similar questions
query = "How can I start investing with little money?"
results = qa_system.search_similar_questions(query, top_k=5)

for idx, (question, similarity) in enumerate(results):
    print(f"{idx+1}. {question} (similarity: {similarity:.4f})")
```

## 🧪 Methodology

### Model Architecture
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Fine-tuning**: Triplet loss with margin α
- **Output**: 384-dimensional sentence embeddings

### Training Details
- **Optimizer**: AdamW with learning rate 2×10⁻⁵
- **Training Data**: 72,810 triplets from Quora dataset
- **Epochs**: 1 full epoch
- **Hardware**: CUDA-enabled GPU

### Evaluation
- **Knowledge Base**: 297,750 unique questions
- **Test Queries**: 44,663 queries
- **Metrics**: MRR, Recall@k, Precision@k, NDCG@k, MAP@k

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for the excellent sentence embedding framework
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Quora](https://www.quora.com/) for providing the Question Pairs dataset
- The Transformers community for advancing NLP research


