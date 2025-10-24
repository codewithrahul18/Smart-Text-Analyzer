# Smart Text Analyzer Wiki

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Smart Text Analyzer** is a comprehensive Natural Language Processing (NLP) tool designed for advanced text analysis using artificial intelligence and machine learning. Built with Python and powered by state-of-the-art NLP libraries, it provides researchers, developers, and data scientists with a robust platform for analyzing text data.

### Key Capabilities
- Real-time sentiment analysis with multiple algorithms
- Named Entity Recognition (NER) using spaCy
- Extractive and abstractive text summarization
- Interactive visualizations and word clouds
- Topic modeling with Latent Dirichlet Allocation (LDA)
- Zero-shot text classification
- PDF report generation
- Text complexity analysis

### Technology Stack
- **Python 3.8+**
- **Transformers** (Hugging Face)
- **spaCy** for NLP processing
- **PyTorch** for deep learning
- **Plotly & Matplotlib** for visualizations
- **NLTK** for natural language processing
- **TextBlob** for sentiment analysis
- **Jupyter/IPython** for interactive environment

---

## Features

### 📊 Basic Text Statistics
Comprehensive analysis of text metrics including character count, word count, sentence metrics, lexical diversity, and estimated reading time.

### 😊 Sentiment Analysis
Dual-model sentiment detection combining TextBlob polarity/subjectivity scores with transformer-based classification for robust consensus analysis.

### 🏷️ Named Entity Recognition
Extract and classify entities (PERSON, ORG, LOCATION, etc.) using spaCy's pre-trained models with detailed entity statistics.

### 📝 Text Summarization
Both extractive summarization (using TF-IDF) and abstractive summarization (using BART models) to condense text while preserving key information.

### 📈 Data Visualization
Generate beautiful, publication-ready visualizations including:
- Word frequency distributions
- Word clouds with customizable parameters
- Sentiment gauge charts
- Entity statistics charts
- Classification confidence charts

### 🧠 Advanced Analysis
- Topic modeling using Latent Dirichlet Allocation
- Zero-shot classification into custom categories
- Text complexity scoring and readability levels
- Part-of-speech tag distribution analysis

### 📄 Report Export
Generate professional PDF reports containing comprehensive analysis results with metadata and timestamps.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB minimum disk space for model downloads

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/smart-text-analyzer.git
   cd smart-text-analyzer
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Models**
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords averaged_perceptron_tagger punkt_tab
   ```

5. **Verify Installation**
   ```bash
   jupyter notebook smart_text_analyzer.ipynb
   ```

### Dependencies
```
transformers>=4.25.0
torch>=2.0.0
spacy>=3.5.0
textblob>=0.17.1
wordcloud>=1.9.0
plotly>=5.13.0
nltk>=3.8.1
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
pandas>=1.5.0
numpy>=1.23.0
fpdf>=1.7.2
reportlab>=4.0.0
gtts>=2.3.0
pygame>=2.1.0
gensim>=4.3.0
ipywidgets>=8.0.0
```

---

## Quick Start

### Basic Usage

```python
from smart_text_analyzer import BasicTextAnalyzer, SentimentAnalyzer

# Initialize analyzer
text = "Your text here..."
analyzer = BasicTextAnalyzer(text)

# Get basic statistics
stats = analyzer.get_basic_stats()
print(f"Word Count: {stats['word_count']}")
print(f"Reading Time: {stats['reading_time_minutes']:.2f} minutes")

# Sentiment analysis
sentiment_analyzer = SentimentAnalyzer()
sentiment = sentiment_analyzer.get_detailed_sentiment(text)
print(f"Consensus: {sentiment['consensus']}")
```

### Using the Jupyter Interface

1. Open the notebook in Jupyter
2. Select text input method (manual or sample)
3. Enter or select text for analysis
4. Click "Analyze Text" button
5. Explore results across multiple tabs

---

## Usage Guide

### Text Input Methods

#### Method 1: Direct Text Entry
Paste or type your text directly into the input field. Supports any length text.

#### Method 2: Sample Texts
Choose from pre-loaded sample texts:
- Product Reviews
- News Articles
- Technical Documents
- Business Reports

### Analysis Tabs

#### 1. Basic Statistics Tab
Displays comprehensive text metrics:
- Character and word counts
- Sentence metrics
- Lexical diversity
- Reading time estimation
- Part-of-speech distribution

#### 2. Sentiment Analysis Tab
Shows sentiment scores from multiple algorithms:
- TextBlob polarity and subjectivity
- Transformer model classification
- Consensus sentiment determination
- Visual sentiment gauges

#### 3. Entities Tab
Lists extracted named entities with:
- Entity text and classification
- Entity type distribution
- Statistics visualization
- Grouped entity display

#### 4. Summary Tab
Provides text summaries:
- Extractive summary (original sentences)
- Abstractive summary (generated sentences)
- Compression statistics
- Adjustable summary length

#### 5. Visualizations Tab
Interactive charts and graphs:
- Word frequency bar charts
- Word clouds
- Entity type distributions
- Classification results

#### 6. Advanced Analysis Tab
- Topic modeling results
- Text complexity metrics
- Zero-shot classification
- Readability assessment

### Export Options

#### Comprehensive Report
Generates formatted text report with:
- Basic statistics summary
- Sentiment analysis results
- Entity information
- Word frequency analysis

#### PDF Export
Creates professional PDF document with:
- Generated timestamp
- Complete analysis metrics
- Entity information
- Word frequency data
- Creator attribution

---

## API Reference

### BasicTextAnalyzer

#### `__init__(text)`
Initialize analyzer with text.

#### `get_basic_stats()`
Returns dictionary with text statistics.

**Returns:**
```python
{
    'char_count': int,
    'word_count': int,
    'sentence_count': int,
    'avg_word_length': float,
    'avg_sentence_length': float,
    'unique_words': int,
    'lexical_diversity': float,
    'paragraph_count': int,
    'reading_time_minutes': float
}
```

#### `get_word_frequency(top_n=20)`
Returns most frequent words with occurrence counts.

**Parameters:**
- `top_n` (int): Number of top words to return

**Returns:**
```python
{'word1': count1, 'word2': count2, ...}
```

#### `get_pos_tags()`
Returns part-of-speech tag distribution.

**Returns:**
```python
{'NN': count, 'VB': count, ...}
```

### SentimentAnalyzer

#### `analyze_with_textblob(text)`
Analyzes sentiment using TextBlob algorithm.

**Returns:**
```python
{
    'polarity': float,      # Range: -1 to 1
    'subjectivity': float,  # Range: 0 to 1
    'sentiment': str        # 'positive', 'negative', 'neutral'
}
```

#### `analyze_with_transformers(text)`
Analyzes sentiment using transformer models.

**Returns:**
```python
{
    'label': str,           # 'POSITIVE' or 'NEGATIVE'
    'score': float,         # Confidence: 0 to 1
    'sentiment': str
}
```

#### `get_detailed_sentiment(text)`
Combines both algorithms for consensus analysis.

**Returns:**
```python
{
    'textblob': dict,
    'transformers': dict,
    'consensus': str        # 'positive', 'negative', 'neutral'
}
```

### EntityExtractor

#### `extract_entities_spacy(text)`
Extracts named entities using spaCy.

**Returns:**
```python
[
    {'text': str, 'label': str, 'start': int, 'end': int},
    ...
]
```

#### `get_entity_summary(text)`
Comprehensive entity extraction and grouping.

**Returns:**
```python
{
    'spacy_entities': list,
    'entity_summary': {'PERSON': [...], 'ORG': [...], ...}
}
```

### TextSummarizer

#### `extractive_summarization(text, num_sentences=3)`
Extracts important sentences using TF-IDF.

**Parameters:**
- `text` (str): Input text
- `num_sentences` (int): Number of sentences in summary

**Returns:** Summarized text (str)

#### `abstractive_summarization(text, max_length=150, min_length=30)`
Generates abstractive summary using BART model.

**Parameters:**
- `text` (str): Input text
- `max_length` (int): Maximum summary length
- `min_length` (int): Minimum summary length

**Returns:** Generated summary (str)

#### `summarize_text(text, method='abstractive')`
Unified summarization method.

**Parameters:**
- `method` (str): 'extractive' or 'abstractive'

**Returns:** Summary (str)

### TopicModeler

#### `extract_topics(text, num_topics=3, num_words=5)`
Extracts topics using Latent Dirichlet Allocation.

**Parameters:**
- `text` (str): Input text
- `num_topics` (int): Number of topics to extract
- `num_words` (int): Words per topic

**Returns:** List of topic descriptions (list)

---

## Configuration

### Model Selection
Configure which models to use in the analysis by modifying model selection in sentimen and summarization classes.

### Visualization Settings
Customize visualizations by adjusting matplotlib style and seaborn palette:
```python
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
```

### NLTK Data
Download additional NLTK data as needed:
```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

### PDF Export Settings
Configure PDF report generation:
```python
pdf_generator = PDFReportGenerator()
# Customize font, colors, and layout in generate_comprehensive_report method
```

---

## Troubleshooting

### Common Issues

#### Issue: spaCy Model Not Found
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

#### Issue: CUDA/GPU Not Detected
**Solution:**
The tool defaults to CPU. For GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: Memory Error with Large Text
**Solution:**
- Truncate text to under 5000 characters for some models
- Use extractive summarization instead of abstractive
- Increase available system memory

#### Issue: Transformers Model Download Timeout
**Solution:**
```python
# Set custom cache directory
import os
os.environ['HF_HOME'] = '/path/to/custom/cache'
```

#### Issue: PDF Generation Fails
**Solution:**
Ensure temporary file permissions are correct and disk space is available.

### Performance Optimization

- Use GPU acceleration for faster processing
- Cache model weights to reduce download time
- Process large texts in batches
- Disable unused analysis modules

---

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where applicable
- Add unit tests for new features

### Reporting Issues

Include the following when reporting bugs:
- Python version
- Operating system
- Complete error message
- Steps to reproduce
- Sample text that causes the issue

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

### Citation

If you use this tool in your research, please cite:

```bibtex
@software{smart_text_analyzer,
  author = {Rahul Chauhan},
  title = {Smart Text Analyzer: Comprehensive NLP Tool},
  year = {2024},
  url = {https://github.com/yourusername/smart-text-analyzer}
}
```

---

## Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check existing documentation

---

**Created by:** Rahul Chauhan  
**Last Updated:** October 2025  
**Version:** 1.0.0
