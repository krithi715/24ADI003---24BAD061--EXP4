#import libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

#load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

#text data cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

df['message'] = df['message'].apply(clean_text)

#vectorization
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])  # spam=1, ham=0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

indices = np.where(y_test != y_pred)[0]

for i in indices[:5]:
    print("Message:", df.iloc[i]['message'])
    print("Actual:", encoder.inverse_transform([y_test[i]])[0])
    print("Predicted:", encoder.inverse_transform([y_pred[i]])[0])
    print("------")

# Laplace Smoothing
model_smooth = MultinomialNB(alpha=0.5)
model_smooth.fit(X_train, y_train)

y_pred_smooth = model_smooth.predict(X_test)
print("Accuracy with smoothing:", accuracy_score(y_test, y_pred_smooth))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.title("Confusion Matrix for Multinomial Naïve Bayes", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)
plt.tight_layout()
plt.show()

#Feature Importance (Top Words Influencing Spam)
feature_names = vectorizer.get_feature_names_out()
spam_prob = model.feature_log_prob_[1]

top_spam_indices = np.argsort(spam_prob)[-15:]

top_words = [feature_names[i] for i in top_spam_indices]
top_values = spam_prob[top_spam_indices]

plt.figure(figsize=(8,6))
plt.barh(top_words, top_values)
plt.title("Top 15 Words Influencing Spam Classification", fontsize=14)
plt.xlabel("Log Probability", fontsize=12)
plt.ylabel("Words", fontsize=12)
plt.tight_layout()
plt.show()

#Word Frequency Comparison (Spam vs Ham - Combined Plot)

spam_messages = df[df['label'] == 'spam']['message']
ham_messages = df[df['label'] == 'ham']['message']

spam_vectorizer = CountVectorizer(stop_words='english')
ham_vectorizer = CountVectorizer(stop_words='english')

spam_vec = spam_vectorizer.fit_transform(spam_messages)
ham_vec = ham_vectorizer.fit_transform(ham_messages)

spam_sum = np.array(spam_vec.sum(axis=0)).flatten()
ham_sum = np.array(ham_vec.sum(axis=0)).flatten()

spam_words = np.array(spam_vectorizer.get_feature_names_out())
ham_words = np.array(ham_vectorizer.get_feature_names_out())

top_spam_indices = spam_sum.argsort()[-10:]
top_ham_indices = ham_sum.argsort()[-10:]

top_spam_words = spam_words[top_spam_indices]
top_spam_counts = spam_sum[top_spam_indices]

top_ham_words = ham_words[top_ham_indices]
top_ham_counts = ham_sum[top_ham_indices]

spam_df = pd.DataFrame({
    'Word': top_spam_words,
    'Frequency': top_spam_counts,
    'Category': 'Spam'
})

ham_df = pd.DataFrame({
    'Word': top_ham_words,
    'Frequency': top_ham_counts,
    'Category': 'Ham'
})

combined_df = pd.concat([spam_df, ham_df])

plt.figure(figsize=(10,8))
sns.barplot(
    data=combined_df,
    x='Frequency',
    y='Word',
    hue='Category'
)

plt.title("Top 10 Most Frequent Words in Spam vs Ham Messages", fontsize=14)
plt.xlabel("Word Frequency", fontsize=12)
plt.ylabel("Words", fontsize=12)
plt.legend(title="Message Type")
plt.tight_layout()
plt.show()