import numpy as np
import scipy.spatial

def pairwise_squared_distances(A, B):
    return scipy.spatial.distance.cdist(A, B)**2
def xb(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u**m
    
    d2 = pairwise_squared_distances(x, v)
    v2 = pairwise_squared_distances(v, v)
    
    v2[v2 == 0.0] = np.inf

    return np.sum(um.T*d2)/(n*np.min(v2))
methods = [xb]
targets = "max max min min min min max".split()

def fcm_get_u(x, v, m):
    distances = pairwise_squared_distances(x, v)
    nonzero_distances = np.fmax(distances, np.finfo(np.float64).eps)
    inv_distances = np.reciprocal(nonzero_distances)**(1/(m - 1))
    return inv_distances.T/np.sum(inv_distances, axis=1)

def fcm(x, c, m=2.0, v=None, max_iter=100, error=0.05):
    if v is None: v = x[np.random.randint(x.shape[0], size=c)]
    u = fcm_get_u(x, v, m)
    for iteration in range(max_iter):
        u_old = u
        um = u**m
        v = np.dot(um, x)/np.sum(um, axis=1, keepdims=True)
        u = fcm_get_u(x, v, m)
        if np.linalg.norm(u - u_old) < error: break
    return v
import matplotlib.pyplot as plt
def random_positive_semidefinite_matrix(d):
    Q = np.random.randn(d, d)
    eigvals = np.random.rand(d)
    return Q.T @ np.diag(eigvals) @ Q
    
    while True:
        A = np.random.rand(d, d)
        A += A.T
        if np.all(np.linalg.eigvals(A) > 0):
            return A
def make_spiral_clusters(c, cluster_size, n_noise, d=2):
    angle = np.linspace(0, 2*np.pi, c, endpoint=False)
    radius = np.linspace(10, 30, c)
    vx = np.cos(angle)*radius
    vy = np.sin(angle)*radius
    v = np.stack([vx, vy], axis=1)

    covariances = np.array([random_positive_semidefinite_matrix(d) for _ in range(c)])

    x = np.concatenate([np.random.multivariate_normal(v[i], covariances[i], cluster_size)
        for i in range(c)], axis=0)

    u = np.random.rand(n_noise, d)
    noise = np.min(x, axis=0)*u + (1 - u)*np.max(x, axis=0)

    x = np.concatenate([x, noise], axis=0)

    return x, v

# generate some data with known number of clusters and some noise
m = 2.0
c_true = 7
x, v_true = make_spiral_clusters(c_true, 2000, 500)

results = []

# cluster data for different number of clusters
cs = np.arange(2, 10)
for c in cs:
    v = fcm(x, c)

    # calculate cluster validity indices
    results.append([])
    for method in methods:
        u = fcm_get_u(x, v, m)

        result = method(x, u, v, m)

        results[-1].append(result)

results = np.array(results)

ny = 4
nx = 2

# plot cluster validity indices
for i, method in enumerate(methods):
    plt.subplot(ny, nx, 1 + i)
    column = results[:, i]
    plt.plot(cs, column)

    # find best cluster size for cluster validity index
    if targets[i] == "min":
        c = cs[np.argmin(column)]
    else:
        c = cs[np.argmax(column)]
    
    plt.title("%s, %s is at %d"%(method.__name__, targets[i], c))
    
    plt.plot([c, c], [np.min(column), np.max(column)])



