# %%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib
import numpy as np
# %%


# Read in data
data = pd.read_csv('./profanity_check/data/clean_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# %%
# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(texts)

# %%

# Train the model
model = LinearSVC(class_weight="balanced", dual=False,
                  tol=1e-2, max_iter=int(1e5))
cclf = CalibratedClassifierCV(estimator=model)
cclf.fit(X, y)


# %%
# Save the model
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(cclf, 'model.joblib')

# %%


def _get_profane_prob(prob):
    return prob[1]


def predict_prob(texts):
    return np.apply_along_axis(_get_profane_prob, 1, cclf.predict_proba(vectorizer.transform(texts)))


texts = ["Fuck you"]
print(vectorizer.transform(texts))
predict_prob(texts)
# cclf.predict_(vectorizer.transform(["Fuck you"]))

# %%
print(X[20])
