# Multilingual-Fake-News-Detection-Model

Project Overview:
This project focuses on detecting fake news across multiple languages using a robust deep-learning approach. The core idea is to process and analyze datasets containing news articles in various languages such as English, Hindi, Bengali, Gujarati, Marathi, and Telugu, and classify them as either Fake or True using the XLM-Roberta transformer model.

Key Features:
1. Multi-language Support: Handles datasets in multiple languages and processes them efficiently.
2. Data Preprocessing: Comprehensive text cleaning to ensure high-quality input for the model.
3. Advanced Visualization: Label distributions and dataset characteristics are visualized with detailed bar plots and pie charts.
4. Deep Learning Architecture: Implements a multi-layer XLM-Roberta classifier with dropout layers to prevent overfitting.
5. Training Pipeline: Optimized training loop with learning rate scheduler and performance monitoring.
6. Evaluation: Measures accuracy and loss on validation data for model performance analysis.

Installation Requirements:
To run the project, the following dependencies are required:

pip install transformers
pip install torch
pip install optuna
pip install tqdm
pip install seaborn
pip install matplotlib

Dataset Details:
The project utilizes datasets in six languages:

English: (True.csv)
Hindi: (dataset-merged.csv)
Bengali: (balanced_bn_data.csv)
Gujarati: (gujarati_news.csv)
Marathi: (marathi_news.csv)
Telugu: (telugu_news.csv)

Each dataset contains:
Text Features: The news content.
Labels: 0 (Fake) or 1 (True).

Workflow
The script checks and lists all files in the input directory to ensure availability of datasets.

Data Preprocessing:
1. Removes unwanted characters, URLs, mentions, and hashtags.
2. Retains alphanumeric characters and scripts specific to supported languages (e.g., Hindi, Bengali).
3. Combines title and text into a single feature for English data.

Data Combination:
1. Merges all datasets into a single dataframe for training and analysis.
2. Shuffles the data to ensure randomness.

Visualization:
1. Plots label distributions for individual datasets and the combined dataset.
2. Saves visualizations as .jpg files for future reference.

Model Architecture:
Uses XLM-Roberta, a multilingual transformer model.
Includes additional fully connected layers:
768 → 512 → 384 → 2 architecture with ReLU activations and dropout layers.

Training Loop:
1. Splits the data into training (80%) and validation (20%) sets.
2. Utilizes AdamW optimizer with ReduceLROnPlateau scheduler for dynamic learning rate adjustment.
3. Tracks progress and loss at intervals for both training and validation phases.

Evaluation:
Computes metrics like validation accuracy and loss.
Prints progress during evaluation.

Visualization Examples:
Bar Plot of Label Distribution (All Datasets):
Displays the number of Fake and True labels in each dataset.

Pie Charts of Proportions:
Illustrates the percentage of fake and true labels for each dataset.

Combined Dataset Label Distribution:
Aggregates all datasets to show overall label distribution.


Results:
Training Accuracy: Continues to improve over epochs.
Validation Accuracy: Indicates the generalizability of the model on unseen data.

Future work:
Experiment with additional transformer models (e.g., BERT, RoBERTa) for comparison.
Incorporate more languages and datasets to improve diversity.
Deploy the model as a REST API for real-time news classification.

Credits
Dataset Sources: Collected from Kaggle and open repositories.
Libraries Used: Hugging Face Transformers, PyTorch, Matplotlib, Seaborn, NLTK, Scikit-learn.
Feel free to contribute and enhance this project! 🎉
