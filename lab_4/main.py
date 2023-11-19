import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Зчитуємо дані з файлу
ds = pd.read_excel("./lab4.xlsx")
all_columns = list(ds.columns[1:-1])
n_products = len(all_columns)
n_params = ds.shape[0]

# Ініціалізуємо масив критеріїв та обчислюємо нормовані значення
f = np.zeros((n_params, n_products))
f0 = np.zeros((n_params, n_products))
for product in range(n_products):
    for criterion in range(n_params):
        f[criterion][product] = ds[all_columns[product]][criterion]

# Обчислюємо суми критеріїв та нормуємо їх
sum_f = np.sum(f, axis=1)
for i in range(n_params):
    f0[i] = f[i] / sum_f[i]

# Обчислюємо integro для кожного продукту
G = np.ones(n_params)
G0 = G / np.sum(G)
G_global = G0[0]

integro = np.zeros(n_products)
for i in range(n_params):
    integro += G0[i] * (1 - f0[i]) ** (-1)

# Знаходимо оптимальний продукт
optimal_product = np.argmin(integro)
print('The best product - ', optimal_product + 1)

# Виводимо графіки
x = np.arange(n_products)
plt.figure(figsize=(10, 6))
plt.bar(x, integro, color='#4bb2c5')
plt.xlabel('Products')
plt.ylabel('Integro')
plt.title('Integro for each product')
plt.show()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(n_params):
    xs = np.arange(n_products)
    ys = np.full((n_products,), i)
    zs = f[i]
    ax.bar(xs, zs, zs=i, zdir='y', color=np.random.rand(3, ))

ax.set_xlabel('Products')
ax.set_ylabel('Criterias')
ax.set_zlabel('Values')
plt.title('3D Bar chart for criteria values')
plt.show()
