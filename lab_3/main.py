import numpy as np
import pandas as pd

ds = pd.read_excel("./lab3.xlsx")

all_columns = list(ds.columns[1:-1])

n_products = len(all_columns)
n_params = ds.shape[0]

f = np.zeros((n_params, n_products))
f0 = np.zeros((n_params, n_products))

for product in range(n_products):
    for criterion in range(n_params):
        f[criterion][product] = ds[all_columns[product]][criterion]

mix = ds["Criterion"]

G = np.ones(n_params)
G0 = np.zeros(n_params)
Gnorm = np.sum(G)
for i in range(n_params):
    G0[i] = G[i] / Gnorm
G_global = G0[0]


def voronin(G0):
    Integro = np.zeros(n_products)
    sum_f = np.zeros(n_params)

    for i in range(n_params):
        sum_f[i] = round(sum(f[i]), 2)
    for i in range(n_params):
        for j in range(len(sum_f)):
            f0[i] = f[j] / sum_f[i]
    for i in range(n_params):
        for j in range(n_products):
            Integro[j] += G0 * (1 - f0[i][j]) ** (-1)

    minimum = 10000
    opt = 0

    for i in range(len(Integro)):
        if minimum > Integro[i]:
            minimum = Integro[i]
            opt = i

    for i in range(len(Integro)):
        print(f'Product {i + 1} -', Integro[i])

    print('The best product - ', opt)
    return voronin


voronin(G_global)
