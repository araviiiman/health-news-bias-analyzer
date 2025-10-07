# Analysis Pipeline Guide

This document explains the step-by-step process of the Health News Bias Analyzer pipeline.

## Overview

The analysis follows this sequence:
1. **Data Collection** → 2. **Preprocessing** → 3. **Topic Modeling** → 4. **Emotion Analysis** → 5. **Bias Detection** → 6. **Visualization**

## Pipeline Steps

### Step 1: Data Collection (`scraping.ipynb`)
**Purpose**: Collect health news articles from Indian news outlets

**Input**: News website URLs
**Output**: Raw article data (CSV files)
**Key Features**:
- Scrapes from TOI, NDTV, The Hindu, India Today
- Extracts title, content, date, source, URL
- Handles duplicates and data cleaning

**Note**: The repository already includes pre-collected data, so this step is optional.

### Step 2: Data Preprocessing (`data description.ipynb`)
**Purpose**: Clean and prepare data for analysis

**Input**: Raw CSV files
**Output**: `preprocessed_health_news_bias.csv`
**Process**:
- Text cleaning (remove HTML, special characters)
- Normalization (lowercase, stopwords)
- Metadata extraction
- Weak supervision labeling (misinfo/accurate/neutral)

### Step 3: Topic Modeling (`Topic_modelling.ipynb`)
**Purpose**: Discover main themes in health news articles

**Input**: `preprocessed_health_news_bias.csv`
**Output**: `topic_modeling_output.csv`, `labeled_topics.csv`
**Method**: BERTopic with sentence transformers
**Key Features**:
- Uses UMAP + HDBSCAN clustering
- Generates 10 interpretable topics
- Topics include: fitness, cancer, mental health, diet, etc.

### Step 4: Bias Detection (`rule_based_bias.ipynb`)
**Purpose**: Initial bias detection using rule-based approach

**Input**: Preprocessed data
**Output**: `bias_features_output.csv`
**Method**: Keyword-based weak supervision
**Features**:
- Detects sensational keywords ("miracle", "cure", "no side effects")
- Identifies scientific language ("study", "trial", "research")
- Creates bias labels: neutral, possibly biased, possibly misinformative

### Step 5: Emotion Analysis (`roberta.ipynb`)
**Purpose**: Classify emotional tone of articles

**Input**: `bias_features_output.csv`
**Output**: `framing_emotion_output.csv`
**Method**: RoBERTa fine-tuned on emotion dataset
**Emotions**: anger, joy, optimism, sadness
**Features**:
- Uses `cardiffnlp/twitter-roberta-base-emotion` model
- Provides emotion probabilities for each article
- Analyzes emotion distribution across bias categories

### Step 6: Misinformation Classification (`misinformation_classification.ipynb`)
**Purpose**: Fine-tuned classification for misinformation detection

**Input**: Emotion-labeled data
**Output**: Trained model and predictions
**Method**: DistilBERT fine-tuning
**Features**:
- Binary classification: misinfo vs accurate
- Uses emotion features as additional input
- Achieves high accuracy on health misinformation

### Step 7: Baseline Comparison (`baseline.ipynb`)
**Purpose**: Compare deep learning models with traditional ML

**Input**: Emotion-labeled data
**Output**: Performance metrics and confusion matrices
**Methods**:
- **Baseline**: Logistic Regression + TF-IDF
- **Advanced**: Fine-tuned DistilBERT
**Metrics**: Accuracy, Precision, Recall, F1-score

### Step 8: Visualization (`viz.ipynb`)
**Purpose**: Create visualizations and analysis dashboard

**Input**: All analysis outputs
**Output**: Charts, plots, and insights
**Visualizations**:
- Word clouds of frequent terms
- Emotion distribution charts
- Topic frequency graphs
- Confusion matrices
- Bias analysis plots

## Data Flow

```
Raw Articles → Preprocessing → Topic Modeling
     ↓              ↓              ↓
Combined CSV → Clean Text → Topic Labels
     ↓              ↓              ↓
Bias Detection → Emotion Analysis → Visualization
     ↓              ↓              ↓
Bias Labels → Emotion Scores → Final Insights
```

## Expected Results

After running the complete pipeline:

1. **Topic Analysis**: 10 main health topics identified
2. **Emotion Distribution**: Articles classified by emotional tone
3. **Bias Detection**: Articles labeled for potential bias/misinformation
4. **Performance Metrics**: Model accuracy and comparison results
5. **Visualizations**: Comprehensive analysis dashboard

## File Dependencies

Make sure to run the notebooks in this order:
1. `scraping.ipynb` (optional - data already provided)
2. `Topic_modelling.ipynb`
3. `rule_based_bias.ipynb`
4. `roberta.ipynb`
5. `misinformation_classification.ipynb`
6. `baseline.ipynb`
7. `viz.ipynb`

## Key Insights from Analysis

- **Emotional Framing**: Misinformative articles show higher emotional intensity
- **Topic Patterns**: Common themes include fitness, cancer, mental health, diet
- **Bias Indicators**: Emotionally charged language correlates with biased reporting
- **Model Performance**: Transformer models significantly outperform traditional ML

## Troubleshooting Pipeline Issues

1. **Memory Issues**: Reduce dataset size for testing
2. **Model Loading**: Ensure stable internet for model downloads
3. **File Dependencies**: Run notebooks in correct sequence
4. **Performance**: Use GPU if available for faster processing
