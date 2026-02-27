import numpy as np
from joblib import load

b = load("artifacts/models/stack_logreg_v1.pkl")
model = b["model"]
print(f"Intercept: {model.intercept_}")
print(f"Coefs: {model.coef_}")
print("Any NaN in coefs:", np.isnan(model.coef_).any())
print("Any Inf in coefs:", np.isinf(model.coef_).any())
