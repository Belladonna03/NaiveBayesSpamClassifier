# Spam Detection using Naive Bayes Classifier 📧🔍

## Overview 🌟
This project implements a spam detection system using the Naive Bayes classifier. The goal is to classify text messages as either spam or ham (non-spam) based on their content. The project involves preprocessing the text data, extracting features, and training a Naive Bayes model to make predictions.

## Introduction 🚀
Spam detection is a common problem in natural language processing (NLP). This project uses the Naive Bayes algorithm, which is a probabilistic classifier based on Bayes' theorem. The algorithm is particularly well-suited for text classification tasks due to its simplicity and effectiveness.

## Dataset 📊
The dataset used in this project is a CSV file named `combined_data.csv`. The dataset contains two columns:

- `text`: The text content of the message.
- `label`: The label indicating whether the message is spam (1) or ham (0).

The dataset is loaded using the `pandas` library, and a brief analysis is performed to understand the distribution of spam and ham messages.

### Dataset Distribution
The dataset is balanced with a roughly equal number of spam and ham messages. A pie chart is generated to visualize the distribution:

![Dataset Distribution](/dataset_distribution.png)

## Preprocessing 🧹
The preprocessing steps include:

- **Lowercasing**: Converting all text to lowercase to ensure uniformity.
- **Removing Non-Alphabetic Characters**: Removing any characters that are not alphabetic.
- **Tokenization**: Splitting the text into individual words.
- **Stopword Removal**: Removing common stopwords that do not contribute to the classification.
- **Stemming**: Reducing words to their root form using the Snowball Stemmer.
- **Lemmatization**: Further reducing words to their base or dictionary form using the WordNet Lemmatizer.

## Feature Extraction 🔍
The preprocessed text is converted into a list of words, which are then used to build a vocabulary. The vocabulary is used to calculate the frequency of each word in both spam and ham messages. These frequencies are used to compute the conditional probabilities required for the Naive Bayes classifier.

## Model Training 🚂
The Naive Bayes classifier is trained using the following steps:

- **Calculate Prior Probabilities**: The prior probabilities of spam and ham messages are calculated based on the dataset.
- **Calculate Conditional Probabilities**: The conditional probabilities of each word given spam and ham are calculated using Laplace smoothing to avoid zero probabilities.

## Evaluation 📈
The model's performance is evaluated using the following metrics:

- **Accuracy**: The proportion of correctly classified messages.
- **Confusion Matrix**: A matrix showing the true positives, true negatives, false positives, and false negatives.
- **Precision**: The proportion of true positives among the predicted positives.
- **Recall**: The proportion of true positives among the actual positives.
- **F1 Score**: The harmonic mean of precision and recall.

### Evaluation Results
Here are the results of the model evaluation:

- **Accuracy**: 0.980
- **Confusion Matrix**:
  ```plaintext
  [[38980   558]
   [ 1134 42776]]
  ```
- **Precision**: 0.987
- **Recall**: 0.974
- **F1 Score**: 0.981

## Usage 🛠️
To run the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/spam-detection.git
   cd spam-detection
   ```
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter notebook or Python script** to train the model and evaluate its performance.

## Dependencies 📦
The project requires the following Python libraries:

- `numpy`
- `pandas`
- `nltk`
- `matplotlib`
- `sklearn`

These dependencies can be installed using the `requirements.txt` file provided in the repository.

## License 📄
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
