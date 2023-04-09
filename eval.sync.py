import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# %%
data = pd.read_csv('./profanity_check/data/clean_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# %%

if os.path.exists('vectorizer.joblib'):
    vectorizer = joblib.load('vectorizer.joblib')
else:
    vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
    X = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, 'vectorizer.joblib')
model = joblib.load('model.joblib')

# %%
# idx = 2
# print("Bad" if y[idx] == 1 else "Good")
# print("----")
# print(texts[idx])
eval_texts = ['Hello there, how are you',
              'Lorem Ipsum is simply dummy text of the printing and typesetting industry.',
              '!!!! Click this now!!! -> https://example.com',
              'fuck you',
              'fUcK u',
              'GO TO hElL, you dirty scum']
tokens = vectorizer.encode(eval_texts)
model.predict(tokens)

# %%
df = pd.read_csv("test_set.csv", sep=",")
test_texts = df["Canonical Form 1"].astype(str)
tokens = vectorizer.encode(test_texts)
preds = model.predict(tokens)
false_neg = df["Canonical Form 1"][preds == 0]
true_pos = df["Canonical Form 1"][preds == 1]
print(len(false_neg)/len(true_pos))
print(true_pos[true_pos["Severity Rating"] > 1])
