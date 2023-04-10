# %%
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib
import numpy as np
# %%
# Read in data

data = pd.read_csv('data/train_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# %%
# Vectorize the text
path_vec = "vectorized.npy"
vectorizer = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2')
if not os.path.exists(path_vec):
    X = vectorizer.encode(list(texts), batch_size=256,
                          show_progress_bar=True)
    np.save(path_vec, X)
else:
    X = np.load(path_vec)

# %%

# Train the model
model = LinearSVC(class_weight="balanced", dual=False,
                  tol=1e-2, max_iter=int(1e6))
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)


# %%
# Save the model
joblib.dump(cclf, 'profanity_protector/data/model.joblib')
