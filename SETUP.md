# Setup Guide for Health News Bias Analyzer

This guide will help you set up and run the Health News Bias Analyzer project on your local machine.

## Prerequisites

- Python 3.8 or higher
- Git
- At least 8GB RAM (16GB recommended)
- Optional: NVIDIA GPU with CUDA support for faster training

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/araviiiman/health-news-bias-analyzer.git
cd health-news-bias-analyzer
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models
The project uses several pre-trained models that will be downloaded automatically on first use:
- `cardiffnlp/twitter-roberta-base-emotion` for emotion classification
- `distilbert-base-uncased` for text classification
- `sentence-transformers/all-MiniLM-L6-v2` for topic modeling

## Running the Analysis Pipeline

### Step 1: Data Collection (Optional)
If you want to collect fresh data:
```bash
jupyter notebook scraping.ipynb
```

### Step 2: Topic Modeling
```bash
jupyter notebook Topic_modelling.ipynb
```

### Step 3: Emotion and Bias Analysis
```bash
jupyter notebook roberta.ipynb
```

### Step 4: Baseline Comparison
```bash
jupyter notebook baseline.ipynb
```

### Step 5: Visualization
```bash
jupyter notebook viz.ipynb
```

## Expected Output Files

After running the complete pipeline, you should have:
- `topic_modeling_output.csv` - Topic analysis results
- `framing_emotion_output.csv` - Emotion classification results
- `bias_features_output.csv` - Bias detection results
- Various visualization plots

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in model training
   - Use fewer samples for testing (uncomment the reduction lines in notebooks)
   - Close other applications to free up RAM

2. **CUDA/GPU Issues**
   - The project works on CPU, but will be slower
   - Install CUDA toolkit if you want GPU acceleration
   - Models will automatically use CPU if GPU is not available

3. **Model Download Issues**
   - Ensure stable internet connection
   - Models are downloaded to `~/.cache/huggingface/` by default
   - You can manually download models if needed

### Performance Tips

- Start with smaller datasets (1000 articles) for testing
- Use the baseline model first to verify setup
- Enable GPU acceleration if available
- Close unnecessary applications during training

## File Structure Explanation

- `scraping.ipynb` - Web scraping for news articles
- `Topic_modelling.ipynb` - BERTopic analysis for discovering themes
- `roberta.ipynb` - Emotion and bias classification
- `baseline.ipynb` - Traditional ML baseline comparison
- `viz.ipynb` - Visualization and results analysis
- `*.csv` files - Various output datasets
- `*.xlsx` - Main dataset file

## Getting Help

If you encounter issues:
1. Check the error messages carefully
2. Ensure all dependencies are installed correctly
3. Verify you have sufficient memory and storage
4. Check the GitHub issues page for common problems

## Hardware Requirements

### Minimum Requirements
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 10GB free space
- Time: 2-4 hours for complete pipeline

### Recommended Requirements
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16GB+
- GPU: NVIDIA GTX 1060 or better
- Storage: 20GB+ free space
- Time: 30-60 minutes for complete pipeline
