import pandas as pd
import numpy as np

np.random.seed(42)

# Generate blood pressure (80-180) and cholesterol levels (100-300)
blood_pressure = np.random.randint(80, 180, 200)
cholesterol = np.random.randint(100, 300, 200)

# Generate labels (1 for disease, 0 for no disease)
disease = np.random.choice([0, 1], size=200, p=[0.5, 0.5])

# Create DataFrame
df = pd.DataFrame({'BloodPressure': blood_pressure, 'Cholesterol': cholesterol, 'Disease': disease})

# Save dataset
df.to_csv("data.csv", index=False)

print("Dataset created successfully: data.csv")
