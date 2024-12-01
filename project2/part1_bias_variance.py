import numpy as np

# Q1.4 : Determine best k* values depending on N and sigma at x = 0
Ns = [25,50]
sigmas = [0.0, 0.1, 0.2]

def terms(k, N, sigma):
    bias = (((k**2) - 1 )/(3*(N-1)**2))**2
    var = sigma**2/k
    error = bias + var + sigma**2
    return error

optimal = np.zeros((len(Ns), len(sigmas), 2))
for i_n, n in enumerate(Ns):
    for i_s, sigma in enumerate(sigmas):
        ks = []
        for k in range(1, n):
            error = terms(k, n, sigma)
            ks.append((k, error))

        min_k, min_error = min(ks, key=lambda x: x[1])
        optimal[i_n, i_s] = (min_k, min_error)

print(optimal)