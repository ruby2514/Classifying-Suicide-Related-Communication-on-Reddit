# Classifying Suicide-Related Communication on Reddit

**Presented at Montclair State University Student Research Symposium, 2026**

## Overview

Online posts can reveal early signs of suicidal thoughts, but the signals are often subtle and impossible for moderators to review at scale. This project builds an NLP pipeline to classify Reddit posts into four suicide-risk categories and analyzes linguistic bias across those categories — asking not just *can we classify these posts*, but *are we doing it fairly*?

> **Research Question:** Can TF-IDF models accurately classify Reddit posts across four suicide-related categories, and which features are most informative?

---

## Dataset

- 500 anonymized Reddit mental-health posts (one per unique user)
- Sourced from: [Kaggle – Text Summary with Reddit Mental Health Posts](https://www.kaggle.com/code/shriyapr/text-summary-with-reddit-mental-health-posts/input)
- Labels: `Supportive`, `Ideation`, `Behavior`, `Attempt`, `Indicator`
- Text is short-to-medium length, informal, and conversational
- Class imbalance present: Ideation posts are most frequent; Attempt posts are least frequent

---

## Methods

**Pipeline:**
```
Reddit Posts → Preprocessing → TF-IDF Vectors → Logistic Regression → Risk Category
```

**Steps:**
1. Text preprocessing: cleaning, tokenizing, filtering low-value tokens to reduce bias in averaging
2. TF-IDF vectorization (unigrams + bigrams)
3. Multiclass Logistic Regression classifier trained on custom risk labels
4. Train/validation/test split
5. Bias metric analysis: bias mean and bias TF-IDF scores computed per category

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## Key Findings

- **Language alone can reveal different suicide-risk levels** in Reddit posts
- **Supportive and Ideation posts are easier to classify** than Behavior and Attempt
- **High-risk Attempt posts carry less detectable linguistic bias** — meaning the most dangerous posts are hardest to flag automatically, which is a critical failure mode for crisis detection tools
- **TF-IDF bias scores are near-zero normally distributed**, indicating that TF-IDF weighting partially balances affective signals
- The model most often **confuses Ideation and Indicator**, reflecting real-world overlap between suicidal thoughts and actions

**Top features per category:**
- `Attempt`: attempt, somebody, suicide attempt, failed
- `Behavior`: cutting, hospital, felt, kill im
- `Ideation`: thank, depression, mental depression, mental
- `Indicator`: behavior, hyperactive, hyperactive behavior, haha
- `Supportive`: think, sounds, therapist help, chocolate

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | TF-IDF vectorization, Logistic Regression, evaluation metrics |
| gensim | Word2Vec embeddings for bias analysis |
| matplotlib | Confusion matrix, bar charts |
| seaborn | Bias distribution histograms |

---

## How to Run

```bash
# Install dependencies
pip install gensim scikit-learn pandas numpy matplotlib seaborn

# Run the notebook
colab notebook classifying_suicide_reddit.ipynb
```

---

## Reference

Zirikly, A., Resnik, P., Uzuner, Ö., & Hollingshead, K. (2019). CLPsych 2019 Shared Task: Predicting the Degree of Suicide Risk in Reddit Posts. *Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology*, 1–12.

---

## Author

**Ruby Villalona** — Montclair State University, College of Science and Mathematics & School of Computing  
villalonar1@montclair.edu
