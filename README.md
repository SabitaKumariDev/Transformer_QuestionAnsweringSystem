**Transformer-Based Question Answering System**

**Overview**

This project implements a Question Answering (QA) system using state-of-the-art transformer-based models such as BERT, ALBERT, and T5. The system is designed to handle diverse datasets, including SQuAD, CovidQA, and CovidGQA, enabling accurate extraction of answers from given context passages. By leveraging fine-tuning techniques on pre-trained transformer models, this project demonstrates advancements in natural language understanding and QA performance.

**Features**

**Transformer Models:** Implements BERT, ALBERT, and T5 for question answering tasks.

**Dataset Compatibility:** Supports popular QA datasets such as SQuAD v1.1, CovidQA, and CovidGQA.

**Fine-Tuning:** Enhances pre-trained models with additional layers like Highway Networks and BiLSTMs to optimize performance.

**Evaluation Metrics:** Measures performance using Exact Match (EM) and F1 scores.

**Flexible Predictions:** Allows users to input custom contexts and questions for real-time predictions.

**Datasets**

The project utilizes the following datasets:

**1. SQuAD v1.1:** Contains over 100k question-answer pairs based on Wikipedia articles.

**2. CovidQA:** Built from the Kaggle CORD-19 dataset, focused on COVID-19-related questions.

**3. CovidGQA:** A manually curated dataset with general COVID-19 questions and answers from medical sources.

Example data entries include:
**Context:** "The series primarily takes place in a region called the Final Empire on a world called Scadrial."

**Question:** "Where does the series take place?"

**Answer:** "A region called the Final Empire"【15†source】【17†source】.

**Model Architecture**

**Base Models**

BERT: Bidirectional Encoder Representations from Transformers, pre-trained on large corpora and fine-tuned for QA tasks.

ALBERT: A lightweight version of BERT with reduced parameters for efficient training.

T5: Text-to-Text Transfer Transformer, converting all NLP tasks into a text-to-text format.

**Enhancements**

**Highway Networks:** Refines embeddings from the transformer layers using gating mechanisms.

**BiLSTM Layers:** Captures relationships between context and questions for improved QA performance【16†source】【19†source】.

**Installation**

**Prerequisites**
Python 3.8 or higher

Libraries: TensorFlow, PyTorch, Hugging Face Transformers, and other dependencies listed in requirements.txt

Setup
1. Clone the repository:
   
git clone https://github.com/your-username/transformer-qa-system.git

cd transformer-qa-system

2. Install dependencies:
   
pip install -r requirements.txt

3. Download datasets and place them in the data/ directory.

**Usage**

**Training**

To fine-tune a transformer model on a dataset:

python train.py --model bert --dataset squad --epochs 3

**Evaluation**

To evaluate a trained model on a test set:

python evaluate.py --model bert --dataset covidqa

**Prediction**

python predict.py --model t5 --context "The Earth is round." --question "What is the shape of the Earth?"

**Results**

Performance metrics for different models:

![image](https://github.com/user-attachments/assets/4439f875-94cf-48b5-9e5e-b52d124be1db)

These results showcase the effectiveness of transformer models in handling various QA tasks【16†source】【19†source】.

**Challenges and Solutions**

**1. Handling Unanswerable Questions:**

Strategy: Modified output layers to differentiate between answerable and unanswerable questions.

**2. Dataset Noise:**

Solution: Cleaned datasets using preprocessing techniques to remove irrelevant or incomplete data.

**3. Performance Bottlenecks:**

Approach: Introduced Highway Networks and BiLSTM layers to optimize embeddings and capture contextual relationships【16†source】【19†source】.

**Future Work**

1. Expand the system to support additional datasets and languages.
   
2. Implement real-time API integration for web-based QA applications.
   
3. Explore alternative pre-training objectives for better generalization.

**References**

**1. BERT:** Pre-training of Deep Bidirectional Transformers for Language Understanding

**2. SQuAD:** The Stanford Question Answering Dataset

**3. T5:** Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer


