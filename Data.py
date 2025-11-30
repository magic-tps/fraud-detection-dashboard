import pandas as pd

# Cargar tu dataset
df = pd.read_csv("creditcard.csv")

# Calcular cuántas filas equivalen a 1/3
n = len(df) // 3

# Eliminar 1/3 aleatorio
df_reducido = df.sample(frac=1, random_state=42).iloc[n:]

# Guardar (opcional)
df_reducido.to_csv("creditcard.csv.csv", index=False)

print(df_reducido.shape)
