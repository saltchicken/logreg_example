import pandas as pd
from ctgan import CTGAN

# Sample dataset
data = pd.DataFrame({
    'age': [25, 32, 40, 29, 50, 45, 36, 23, 60, 38],
    'race': ['White', 'Black', 'Asian', 'Hispanic', 'White', 'Black', 'Asian', 'Hispanic', 'White', 'Black'],
    'height_cm': [175, 180, 165, 170, 185, 178, 160, 172, 190, 176],
    'income': [50000, 60000, 55000, 52000, 70000, 65000, 48000, 51000, 75000, 62000]
})

# Define categorical columns
categorical_columns = ['race']

# Train CTGAN model
model = CTGAN(epochs=300)
model.fit(data, categorical_columns)

# Generate 5 synthetic samples
synthetic_data = model.sample(5)

# Show generated data
print(synthetic_data)

