
# Toxic Comment Classifier

A collaborative machine learning project focused on detecting toxic online comments using classical NLP techniques. We explored and benchmarked various models including Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).

> Dataset: [Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

## Team Members

- [**Dwiti**](https://github.com/DwitiThaker)  
- [**Bhavya Mistry**](https://github.com/Bhavya-Mistry)  
- [**manan-25**](https://github.com/manan-25)

---

## Objective

To build a robust and interpretable classifier to detect toxic comments and compare different classical ML models, using clean data pipelines and meaningful evaluation.

---

## Tech Stack

- **Language:** Python 3  
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`  
- **Modeling Techniques:**  
  - Logistic Regression  
  - Support Vector Machines (SVM)  
  - Multinomial Naive Bayes (MNB)  
- **Text Features:**  
  - Unigrams & Bigrams  
  - CountVectorizer with binary flags  

---

## Exploratory Data Analysis (EDA)

- Checked class imbalance across toxic labels  
- Explored comment length distributions  
- Visualized word frequency in toxic vs. non-toxic comments using:
  - Word clouds
  - Bar charts
- Observed correlations between different toxicity labels

---

## Data Preprocessing

- Lowercasing  
- Removal of URLs, special characters, and punctuation  
- Tokenization  
- Binarized bag-of-words vectorization with unigrams and bigrams  

---

## Models Implemented

| Model                  | Description |
|------------------------|-------------|
| **Logistic Regression** | Baseline linear model with L2 regularization |
| **Support Vector Machine (SVM)** | Robust linear classifier using hinge loss |
| **Multinomial Naive Bayes (MNB)** | Probabilistic model for word occurrence |

---

## Results & Evaluation

- Metrics Used: Accuracy, Precision, Recall, F1-score  
- Comparison visualizations and confusion matrices included in the notebook  
- Insights discussed for each model performance

**Logistic Regression** : 91.99 <br>
**SVM** : 91.96 <br>
**Naive Bayes** : 91.32 <br>

> See detailed results in the [Toxic.ipynb](https://github.com/Bhavya-Mistry/Collaborative-ml-workspace/blob/main/toxic-comment-classifier/Code/toxic-comment-classifier.ipynb) notebook.

---

## References

- [Sida Wang & Christopher Manning, 2012. “Baselines and Bigrams: Simple, Good Sentiment and Topic Classification”](https://aclanthology.org/P12-2018/)
- [Jigsaw Toxic Comment Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

## Getting Started

To run this project on your local machine:

```bash
git clone https://github.com/Bhavya-Mistry/Collaborative-ml-workspace/Toxic
cd toxic-comment-classifier
pip install -r requirements.txt
jupyter notebook Toxic.ipynb        
