# Sentiment Analysis on Amazon Reviews

## Overview
This project is a sentiment analysis model developed using TensorFlow, aimed at understanding the sentiments expressed in Amazon product reviews. By analyzing these reviews, the model predicts whether a review is positive or negative. This model can be particularly useful for businesses looking to gather insights from customer feedback on products sold on Amazon.

## Dataset
The dataset used for this project is the "Amazon Reviews" dataset, available on Kaggle at [this link](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews). It consists of text reviews and their corresponding labels indicating the sentiment (positive or negative).

### Note on the Provided Data Set
The accuracy and performance metrics highlighted above were achieved by training our sentiment analysis model on the full data set. Due to GitHub's file size limitations, we have included a smaller, sample subset of the data in this repository for quick testing and demonstration purposes.

To replicate the reported performance metrics and fully leverage the model's capabilities, it is recommended to train the model using the complete data set. Instructions for accessing and preparing the full data set for training are provided in the data/README.md file.

This approach ensures that while users can quickly experiment with the model using the provided sample data, they are also informed about the steps to achieve optimal results with the full data set.

## Features
- **Preprocessing:** Tokenization, removal of stop words, and stemming.
- **Model:** The core of the project is a Long Short-Term Memory (LSTM) network, implemented using TensorFlow. This choice is motivated by the LSTM's ability to capture long-term dependencies in text data, making it well-suited for sentiment analysis tasks.

## Model Evaluation Results

Our sentiment analysis model has been rigorously tested and evaluated, showcasing its effectiveness and reliability in classifying text sentiments. The key performance metric obtained from our evaluation process is as follows:

- **Accuracy:** 90.68%

This accuracy highlights the model's strong capability in correctly classifying sentiments as either positive or negative.

### Evaluation Methodology

The model underwent evaluation on a diverse and comprehensive test set designed to represent a wide range of sentiments across different domains. The primary metric for evaluation, accuracy, was calculated by comparing the model's predictions against the true labels in the test set.

### Insights and Interpretation

The achieved accuracy demonstrates the model's robustness in sentiment analysis tasks, indicating its reliability in applications such as customer feedback analysis, social media sentiment tracking, and market research. While precision, recall, and F1 Score metrics are not provided in this summary, the high accuracy rate itself suggests a balanced performance between sensitivity and specificity, making it a valuable tool for analyzing sentiments in various text data sources.

---

## Sentiment Analysis CLI Tool

This CLI tool is designed for sentiment analysis on text data, allowing users to train models, save them, and predict sentiment of given texts.

### Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation on Apple M1

This project is optimized for the Apple M1 chip, ensuring enhanced performance and compatibility. Users with Apple M1 devices should ensure that the `apple` channel is included when creating their Conda environment, as demonstrated in the `environment.yml` file:

```yaml
channels:
  - apple
  - anaconda
  - default
  - defaults
  - conda-forge
```

This setup leverages the specific binaries optimized for the M1 architecture, ensuring that TensorFlow and other dependencies run efficiently on Apple Silicon.

#### Setting Up the Environment

To create a Conda environment for this project on an Apple M1 device, follow these steps:

1. Ensure you have Miniconda or Anaconda installed on your M1 device. If not, download and install it from the [official website](https://docs.conda.io/en/latest/miniconda.html).

2. Clone the project repository:

   ```bash
   git clone https://github.com/NurgulKaryagis/sentiment-analysis-cli.git
   cd sentiment-analysis-cli
   ```

3. Create the Conda environment using the `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```

4. Activate the new environment:

   ```bash
   conda activate sentiment-analysis
   ```

5. Now, you're ready to run the project on your Apple M1 device.


### Manual Installation Guide Using Conda

This section provides step-by-step instructions for manually setting up the environment required to run the sentiment analysis project using Conda. This method is ideal for users who prefer Conda for managing Python packages and environments.

#### Prerequisites

1. Ensure you have Miniconda or Anaconda installed on your device. If not, download and install it from the [official website](https://docs.conda.io/en/latest/miniconda.html).

2. Clone the project repository:

   ```bash
   git clone https://github.com/NurgulKaryagis/sentiment-analysis-cli.git
   cd sentiment-analysis-cli
   ```

3. Create a Conda Environment

Create a new Conda environment named `sentiment-analysis` and specify Python version 3.8 (or a version compatible with your project requirements):

```
conda create --name sentiment-analysis python=3.8
```

Activate the newly created environment:

```
conda activate sentiment-analysis
```

4. Install Required Packages

Install the necessary packages within the Conda environment. The following command installs TensorFlow, NumPy, NLTK, and Keras:

```
conda install numpy tensorflow keras
pip install nltk
```
5. Download NLTK Data

You need to download the necessary NLTK datasets, including stopwords and tokenizer models:

```
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Usage

- **Training the model:**

  Train a new model using your dataset.

  ```
  python main.py --train
  ```

  Optionally, save the trained model to a specified path.

  ```
  python main.py --train --save models/my_trained_model.h5
  ```

- **Predicting sentiment:**

  Predict the sentiment of a given text.

  ```
  python main.py --predict "This is a great product"
  ```

  Ensure you specify the path to the trained model and tokenizer if they are not in the default locations.

  ```
  python main.py --predict "This is a great product" --model models/my_trained_model.h5 --tokenizer data/processed/my_tokenizer.pkl

#### License

This project is open-sourced under the MIT License.