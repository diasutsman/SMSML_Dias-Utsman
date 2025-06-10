import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create a minimal sklearn model
model = RandomForestClassifier(n_estimators=10)
model.fit(np.array([[5.1, 3.5, 1.4, 0.2]]), [0])

# Save the model
with open('artifacts/model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Created placeholder model.pkl file")
