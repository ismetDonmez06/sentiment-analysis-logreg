import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import Word

# Load Data
data = pd.read_csv("/content/train.tsv", sep="\t")

# Map sentiment labels into binary classes
data["Sentiment"].replace(0, "negative", inplace=True)
data["Sentiment"].replace(1, "negative", inplace=True)
data["Sentiment"].replace(3, "positive", inplace=True)
data["Sentiment"].replace(4, "positive", inplace=True)
data = data[(data.Sentiment == "negative") | (data.Sentiment == "positive")]

df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]

# Text Preprocessing
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
df['text'] = df['text'].str.replace(r'\d', '', regex=True)

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Train-Test Split
train_x, test_x, train_y, test_y = model_selection.train_test_split(
    df["text"], df["label"], random_state=1
)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.transform(test_y)

# Feature Engineering
count_vectorizer = CountVectorizer()
x_train_count = count_vectorizer.fit_transform(train_x)
x_test_count = count_vectorizer.transform(test_x)

tfidf_vectorizer = TfidfVectorizer()
x_train_tfidf = tfidf_vectorizer.fit_transform(train_x)
x_test_tfidf = tfidf_vectorizer.transform(test_x)

# Logistic Regression Model
loj = linear_model.LogisticRegression()

loj.fit(x_train_count, train_y)
y_pred_count = loj.predict(x_test_count)
print("Count Vectors Accuracy:", metrics.accuracy_score(test_y, y_pred_count))

loj.fit(x_train_tfidf, train_y)
y_pred_tfidf = loj.predict(x_test_tfidf)
print("Word-level TF-IDF Accuracy:", metrics.accuracy_score(test_y, y_pred_tfidf))
