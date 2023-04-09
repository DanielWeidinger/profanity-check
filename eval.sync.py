import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# %%
data = pd.read_csv('./test_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# %%

vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

# %%
eval_texts = ['Hello there, how are you',
              'Lorem Ipsum is simply dummy text of the printing and typesetting industry.',
              '!!!! Click this now!!! -> https://example.com',
              'fuck you',
              'fUcK u',
              'GO TO hElL, you dirty scum']
tokens = vectorizer.encode(eval_texts)
model.predict(tokens)

# %%
df = pd.read_csv("user_inputs.csv", sep=",")
test_texts = df["translated"].astype(str)
tokens = vectorizer.encode(test_texts)
preds = model.predict(tokens)


def _get_profane_prob(prob):
    return prob[1]


preds_probs = np.apply_along_axis(
    _get_profane_prob, 1, model.predict_proba(vectorizer.encode(test_texts)))
true_pos = df["translated"][preds == 1]
print(true_pos)
print(preds_probs[preds == 1])

# %%
y_pred = model.predict(vectorizer.encode(texts))

# %%
conf_matrix = confusion_matrix(y, y_pred)
print(f"f1: {f1_score(y, y_pred)}")
print(f"acc: {accuracy_score(y, y_pred)}")
