import numpy as np
from joblib import load

b = load("artifacts/models/stack_logreg_v1.pkl")
model = b["model"]
print("Model classes: ", model.classes_)
print("Coef type: ", model.coef_.dtype)
