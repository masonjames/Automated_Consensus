```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def synthesize_personas(dataframe):
    # Normalize the data
    df_normalized = (dataframe - dataframe.mean()) / dataframe.std()

    # Apply PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_normalized)

    # Create a DataFrame with the principal components
    df_principal = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    # Return the synthesized personas
    return df_principal

# Load the data
df = pd.read_csv('data.csv')

# Synthesize personas
personas = synthesize_personas(df)

# Save the personas to a CSV file
personas.to_csv('personas.csv', index=False)
```