# Health News Bias Analyzer

A comprehensive analysis pipeline for detecting media bias and misinformation in Indian health news articles using advanced NLP and deep learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The rise of digital media has dramatically transformed how health information is disseminated. In India, news media plays a critical role as a primary source of public health knowledge. However, the increasing prevalence of biased reporting and misinformation—particularly during health crises—has raised significant concerns about the accuracy and impact of such coverage.

This project addresses these challenges by analyzing linguistic patterns in health news articles from major Indian news organizations using a comprehensive pipeline that includes:

- **Web Scraping**: Automated collection from major Indian news outlets
- **Topic Modeling**: Unsupervised discovery of health themes using BERTopic
- **Emotion Analysis**: Detection of emotional framing using DistilBERT
- **Bias Classification**: Identification of editorial slants using RoBERTa
- **Misinformation Detection**: Supervised classification of potentially misleading content

## ✨ Features

- **Multi-source Data Collection**: Scrapes articles from The Times of India, NDTV, The Hindu, and India Today
- **Advanced Topic Modeling**: Uses BERTopic with sentence transformers for semantic topic discovery
- **Emotion Classification**: Detects anger, joy, optimism, and sadness in health articles
- **Bias Detection**: Identifies framing bias and emotional manipulation
- **Deep Learning Pipeline**: Fine-tuned transformer models for classification tasks
- **Comprehensive Visualization**: Interactive charts and analysis dashboards

## 📊 Dataset

- **Size**: 50,000+ unique health-related news articles
- **Sources**: Major Indian news outlets (TOI, NDTV, The Hindu, India Today)
- **Language**: English articles focusing on health topics
- **Time Period**: Articles collected over multiple months
- **Structure**: Each record includes title, content, source, date, and URL

### Dataset Statistics
- **Vocabulary Size**: 64,509 unique words after preprocessing
- **Top Categories**: Health practices, disease mentions, institutional references
- **Average Article Length**: Variable (health articles typically 200-2000 words)

## 🔬 Methodology

### 1. Data Collection & Preprocessing
- Custom web scraping pipeline using `requests`, `BeautifulSoup`, and `newspaper3k`
- Text cleaning and normalization
- Metadata extraction and standardization

### 2. Topic Modeling (BERTopic)
- Uses sentence-transformer embeddings for semantic similarity
- Combines UMAP dimensionality reduction with HDBSCAN clustering
- Generates interpretable topic clusters for health themes

### 3. Emotion Classification (DistilBERT)
- Fine-tuned on GoEmotions dataset
- Classifies articles into anger, joy, optimism, sadness
- Achieves 86.29% accuracy with 0.74 macro F1-score

### 4. Bias Detection (RoBERTa)
- Identifies editorial slants and framing bias
- Detects fear appeals, sensationalism, and authority framing
- Provides emotion distribution analysis across bias categories

### 5. Visualization & Analysis
- Word clouds for dominant vocabulary themes
- Emotion and framing distribution charts
- Topic frequency analysis
- Model performance visualizations

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/araviiiman/health-news-bias-analyzer.git
cd health-news-bias-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models** (optional, will be downloaded automatically on first use)
```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-emotion')"
```

## 💻 Usage

### Quick Start

For a comprehensive example, start with:
```python
# Run the example notebook to see the complete workflow
jupyter notebook EXAMPLE_USAGE.ipynb
```

### Detailed Setup

For detailed installation and setup instructions, see:
- **[SETUP.md](SETUP.md)** - Complete installation guide
- **[PIPELINE.md](PIPELINE.md)** - Step-by-step pipeline explanation

### Pipeline Execution

The complete analysis pipeline can be run in sequence:

1. `scraping.ipynb` - Data collection and preprocessing
2. `Topic_modelling.ipynb` - Topic discovery and analysis
3. `rule_based_bias.ipynb` - Initial bias detection
4. `roberta.ipynb` - Advanced emotion and bias classification
5. `baseline.ipynb` - Traditional ML baseline comparison
6. `viz.ipynb` - Visualization and results analysis

### Individual Components

**Data Collection**
```python
# Run the scraping notebook
jupyter notebook scraping.ipynb
```

**Topic Modeling**
```python
# Analyze topics in health articles
jupyter notebook Topic_modelling.ipynb
```

**Emotion Analysis**
```python
# Classify emotions in articles
jupyter notebook roberta.ipynb
```

**Baseline Comparison**
```python
# Compare with traditional ML approaches
jupyter notebook baseline.ipynb
```

## 📈 Results

### Model Performance

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Logistic Regression (TF-IDF) | 80.89% | 0.51 | 0.79 |
| **DistilBERT (Fine-tuned)** | **86.29%** | **0.74** | **0.86** |

### Emotion Classification Results

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Anger | 0.61 | 0.69 | 0.65 | 16 |
| Joy | 0.80 | 0.78 | 0.79 | 185 |
| Optimism | 0.64 | 0.60 | 0.62 | 42 |
| Sadness | 0.91 | 0.92 | 0.92 | 479 |

### Key Findings

- **Emotional Framing**: Articles labeled as potentially misinformative show higher emotional intensity
- **Topic Patterns**: Common themes include fitness, cancer treatment, mental health, and diet
- **Bias Indicators**: Emotionally charged language often correlates with biased reporting
- **Model Performance**: Transformer-based approaches significantly outperform traditional ML methods

## 📁 Project Structure

```
health-news-bias-analyzer/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── scraping.ipynb                      # Data collection pipeline
├── Topic_modelling.ipynb               # BERTopic analysis
├── rule_based_bias.ipynb               # Initial bias detection
├── roberta.ipynb                       # Emotion & bias classification
├── baseline.ipynb                      # Traditional ML baseline
├── misinformation_classification.ipynb # Misinformation detection
├── sent_anal.ipynb                     # Sentiment analysis
├── viz.ipynb                          # Visualization dashboard
├── data description.ipynb              # Dataset exploration
├── indian_health_bias_combined_clean.xlsx  # Main dataset
├── bias_features_output.csv            # Bias analysis results
├── framing_emotion_output.csv          # Emotion classification results
├── topic_modeling_output.csv           # Topic modeling results
├── labeled_misinfo_data.csv            # Labeled misinformation data
└── labeled_topics.csv                  # Topic labels
```

## 🛠 Technologies Used

### Core Libraries
- **Transformers**: HuggingFace library for BERT, RoBERTa, DistilBERT
- **BERTopic**: Advanced topic modeling with transformer embeddings
- **Scikit-learn**: Traditional ML algorithms and evaluation metrics
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Web Scraping
- **BeautifulSoup**: HTML parsing
- **Requests**: HTTP requests
- **Newspaper3k**: Article extraction

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **WordCloud**: Text visualization

### Deep Learning
- **PyTorch**: Model training and inference
- **Transformers**: Pre-trained model loading
- **Sentence-Transformers**: Embedding generation

## 🔧 Hardware Requirements

### Minimum
- CPU: Intel i5 or equivalent
- RAM: 8GB
- Storage: 10GB free space

### Recommended
- CPU: Intel i7 or equivalent
- RAM: 16GB+
- GPU: NVIDIA GeForce GTX 1060 or better
- Storage: 20GB+ free space

## 🚀 Future Enhancements

- **Multi-task Learning**: Joint emotion and bias classification
- **Multilingual Support**: Regional language analysis
- **Real-time Monitoring**: Live bias detection dashboard
- **Expert Annotations**: Human-validated training data
- **Multimodal Analysis**: Image and video content analysis
- **Explainable AI**: Model interpretability tools

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{health-news-bias-analyzer,
  title={Analyzing Media Bias in Health Reporting: Scraping News Articles for Language Patterns in Health Misinformation},
  author={Aravind},
  year={2024},
  url={https://github.com/araviiiman/health-news-bias-analyzer}
}
```

## 📞 Contact

- **Author**: Aravind
- **GitHub**: [@araviiiman](https://github.com/araviiiman)
- **Project Link**: [https://github.com/araviiiman/health-news-bias-analyzer](https://github.com/araviiiman/health-news-bias-analyzer)

---

⭐ If you found this project helpful, please give it a star!