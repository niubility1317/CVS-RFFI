"""The published 15-neighbor Euclidean majority-vote rule."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def classify(enrollment, enrollment_labels, query, *, neighbors=15):
    enrollment, enrollment_labels, query = map(np.asarray, (enrollment, enrollment_labels, query))
    if enrollment.ndim != 2 or query.ndim != 2 or enrollment.shape[1] != query.shape[1]:
        raise ValueError('enrollment and query embeddings must have matching dimensions')
    if enrollment_labels.ndim != 1 or len(enrollment_labels) != len(enrollment):
        raise ValueError('enrollment labels must match enrollment embeddings')
    if neighbors <= 0 or len(enrollment) < neighbors or not len(query):
        raise ValueError('insufficient enrollment or query records')
    if not np.isfinite(enrollment).all() or not np.isfinite(query).all():
        raise ValueError('embeddings must be finite')
    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
    classifier.fit(enrollment, enrollment_labels)
    return classifier.predict(query)
