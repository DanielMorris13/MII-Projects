# Importa le librerie necessarie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm, norm, expon, poisson
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

# Caricamento del dataset (sostituisci 'danish.csv' con il percorso del tuo file)
L = pd.read_csv('danish.csv')['Loss']  # Assumendo che il dataset si chiami 'danish.csv'

# Per esempio, possiamo caricare un dataset fittizio
# Supponiamo che `danish.Loss` sia una lista di valori numerici
L = np.random.gamma(2, 100, size=1000)  # Esempio di dati di perdita simulati

# Ispezione dei dati
M = np.mean(L)
S = np.std(L)

# Creare un istogramma delle perdite
plt.figure()
plt.hist(L, bins=30, density=True)
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.title('Histogram of Loss Data')
plt.show()

# Distribuzione empirica
x = np.sort(L)
F = np.arange(1, len(x) + 1) / len(x)

# Grafico della distribuzione empirica
plt.figure()
plt.scatter(np.log(x), F)
plt.xlabel('Logged claim sizes')
plt.ylabel('Empirical Distribution')
plt.title('Empirical Distribution')
plt.show()

# Indice delle dimensioni delle perdite
Ix = np.zeros(len(x))
E = np.zeros(len(x))
for i in range(len(x)):
    Ind = x <= x[i]
    Ix[i] = np.mean(x * Ind) / np.mean(x)
    E[i] = np.mean((x - x[i]) * (x > x[i]))

# Grafico dell'indice delle dimensioni delle perdite
plt.figure()
plt.scatter(F, Ix)
plt.xlabel('Number of claims (%)')
plt.ylabel('Empirical Loss Size Index')
plt.title('Empirical Loss Size Index')
plt.show()

# Funzione mean Excess
plt.figure()
plt.scatter(x, E)
plt.xlabel('Threshold')
plt.ylabel('Mean Excess Function')
plt.title('Mean Excess Function')
plt.show()

# Densità Kernel
kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(L[:, None])
X_dens = np.linspace(min(L), max(L), 1000)[:, None]
log_dens = kde.score_samples(X_dens)
plt.figure()
plt.plot(X_dens, np.exp(log_dens))
plt.title('Kernel Density')
plt.show()

# Estimazione delle distribuzioni

# Metodo dei momenti per Gamma
theta = S**2 / M
k = M / theta

# Maximum Likelihood
from scipy.stats import gamma, lognorm, norm

# Estimazione Gamma MLE
params_gamma = gamma.fit(L, floc=0)
LL_Gamma = np.sum(np.log(gamma.pdf(L, *params_gamma)))

# Estimazione Lognorm MLE
params_lognorm = lognorm.fit(L, floc=0)
LL_Logn = np.sum(np.log(lognorm.pdf(L, *params_lognorm)))

# Estimazione Normale MLE
params_norm = norm.fit(L)
LL_Norm = np.sum(np.log(norm.pdf(L, *params_norm)))

# Traccia le distribuzioni stimate
x_vals = np.linspace(min(L), max(L), 1000)
plt.figure()
plt.plot(x_vals, gamma.pdf(x_vals, *params_gamma), label='Gamma MLE')
plt.plot(x_vals, lognorm.pdf(x_vals, *params_lognorm), label='LogNormal MLE')
plt.plot(x_vals, norm.pdf(x_vals, *params_norm), label='Normal MLE')
plt.xlabel('Loss')
plt.ylabel('Density')
plt.title('Probability Density Functions')
plt.legend()
plt.show()

# Test di Kolmogorov-Smirnov
from scipy.stats import kstest

h1 = kstest(L, 'gamma', args=params_gamma)
h2 = kstest(L, 'lognorm', args=params_lognorm)
h3 = kstest(L, 'norm', args=params_norm)

# AIC e BIC
AIC_Gamma = 2 * 2 - 2 * LL_Gamma
AIC_Logn = 2 * 2 - 2 * LL_Logn
AIC_Norm = 2 * 2 - 2 * LL_Norm

BIC_Gamma = 2 * np.log(len(L)) - 2 * LL_Gamma
BIC_Logn = 2 * np.log(len(L)) - 2 * LL_Logn
BIC_Norm = 2 * np.log(len(L)) - 2 * LL_Norm


# Simulazione Monte Carlo per il portafoglio di polizze
lambda_poisson = 2
num_policies = 10000

Nsin = poisson.rvs(lambda_poisson, size=num_policies)
Loss = np.zeros(num_policies)

for i in range(num_policies):
    Loss[i] = np.sum(lognorm.rvs(params_lognorm[0], params_lognorm[1], size=Nsin[i]))


# Esempio di calcolo del premio puro usando il principio della varianza
Vt = lambda_poisson * np.exp(2 * params_lognorm[0] + params_lognorm[1]**2) * (np.exp(params_lognorm[1]**2) - 1) + lambda_poisson * (np.exp(params_lognorm[0] + params_lognorm[1]**2 / 2))**2
Vare = np.var(Loss)
car_var = 0.00001
Premiot_var = (1 + car_var) * Vt
Premioe_var = (1 + car_var) * Vare

# Grafico ECDF
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(Loss)
plt.figure()
plt.scatter(ecdf.x, ecdf.y)
plt.title('Empirical CDF')
plt.show()

# Metodo Bootstrap
Lossb = np.zeros(num_policies)
for i in range(num_policies):
    Lossb[i] = np.sum(np.random.choice(L, Nsin[i], replace=True))

# Visualizzazione della CDF bootstrap
ecdf_b = ECDF(Lossb)
plt.figure()
plt.scatter(ecdf_b.x, ecdf_b.y)
plt.title('Bootstrap CDF')
plt.show()
