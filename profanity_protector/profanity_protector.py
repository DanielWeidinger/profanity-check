import pkg_resources
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

vectorizer = SentenceTransformer(pkg_resources.resource_filename('profanity_protector',
                                                                 'data/sentence-transformers_all-MiniLM-L6-v2'))
model = joblib.load(pkg_resources.resource_filename(
    'profanity_protector', 'data/model.joblib'))


def _get_profane_prob(prob):
    return prob[1]


def predict(texts):
    return model.predict(vectorizer.encode(texts))


def predict_prob(texts):
    return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.encode(texts)))
