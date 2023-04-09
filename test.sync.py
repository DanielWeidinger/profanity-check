# %%
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib
import numpy as np
import torch
# %%

# Read in data
data = pd.read_csv('./profanity_check/data/clean_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# batch_size = 64
# num_rows = data.shape[0]
# num_cols = data.shape[1]
# texts = texts.iloc[:len(data)-(num_rows % batch_size)]
# y = y.iloc[:len(data)-(num_rows % batch_size)]
#
# new_shape = (-1, batch_size, num_cols)
# texts = np.array(texts).reshape(new_shape)
# y = np.array(y).reshape(new_shape)
# print(y.shape)
#
# print(f"Count datapoints: {len(data)}")

# %%
# Vectorize the text
vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
X = vectorizer.encode(texts, batch_size=256,
                      show_progress_bar=True)
np.save("vectorized.npy", X)


# vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
# X = vectorizer.fit_transform(texts)

# %%

# Train the model
model = LinearSVC(class_weight="balanced", dual=False,
                  tol=1e-2, max_iter=int(1e5))
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)


# %%
# Save the model
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(cclf, 'model.joblib')

# %%


def _get_profane_prob(prob):
    return prob[1]


def predict_prob(texts):
    return np.apply_along_axis(_get_profane_prob, 1, cclf.predict_proba(vectorizer.encode(texts)))


texts = ["Fuck you"]
predict_prob(texts)
# cclf.predict_(vectorizer.transform(["Fuck you"]))

# %%
print(X[20])
