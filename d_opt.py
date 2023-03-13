import numpy as np
import matplotlib.pyplot as plt

from particle import Swarm
from matplotlib.lines import Line2D

np.random.seed(123456)


def h(z):
    _z = np.exp(z)
    return _z / (1 + _z)


def h_prime(z):
    _z = np.exp(z)
    return _z / ((1 + _z) ** 2)


def z(x, a, b, c):
    return a + b * x + c * x ** 2


def dzda(x, a, b, c):
    return 1.0


def dzdb(x, a, b, c):
    return x


def dzdc(x, a, b, c):
    return x ** 2


def Mx(x, a, b, c):
    _z = z(x, a, b, c)
    _h = h(_z)
    f = np.array([dzda(x, a, b, c), dzdb(x, a, b, c), dzdc(x, a, b, c)]).reshape(3, 1)
    f = np.dot(f, f.T)
    f = f * h_prime(_z) ** 2.0
    Mx = f / (_h * (1.0 - _h))
    return Mx


def convert_W_to_wis(W):
    return np.append(W, 1 - np.sum(W))


def Mw(xis, wis, a, b, c):
    _Mw = np.zeros((3, 3))
    for x, w in zip(xis, wis):
        _Mw += Mx(x, a, b, c) * w
    return _Mw


def detM(M):
    return np.linalg.det(M)


def logdetM(M):
    return np.log(np.linalg.det(M))


def logdetMdm(M):
    inv = np.linalg.inv(M)
    if check_symmetric(M):
        return inv
    return 2 * inv - np.diag(inv)


def efficiency(detM, detM_opt, k):
    return (detM / detM_opt) ** (1 / k)


def check_symmetric(a, tol=1e-6):
    return np.all(np.abs(a - a.T) < tol)


a, b, c = -9.0, 3.0, -0.214
uniform_X = np.linspace(0, 10, 30)
uniform_W = np.ones(29) / 30
small_X = np.linspace(6.5, 7.5, 3)
small_W = np.ones(2) / 3
query_doses = np.linspace(0, 10, 101)
response = h(z(query_doses, a, b, c))


def fitness(X, W):
    wis = convert_W_to_wis(W)
    _Mw = Mw(X, wis, a, b, c)
    return detM(_Mw)


s = Swarm(20, 3, 10.0, 0.0, fitness)
s.optimise(1000, True)

_Mw = Mw(s.X_best, s.W_best, a, b, c)

print(f'Optimal doses: {s.X_best}, Optimal weights {s.W_best}')
print(f'For D-Opt: Efficiency D: {1.0}')
print(f'For uniform: Efficiency D: {efficiency(fitness(uniform_X, uniform_W), s.global_fitness_best, k=3)}')
print(f'For three near optimal: Efficiency D: {efficiency(fitness(small_X, small_W), s.global_fitness_best, k=3)}')

fig, ax = plt.subplots()
ax.plot(query_doses, response, color='black')
ax.set_xlabel("Dose")
ax.set_ylabel("Efficacy probability")
ax2 = ax.twinx()
rs = ax2.scatter(s.X_best, convert_W_to_wis(s.W_best), color="red", marker="o")
rb = ax2.scatter(uniform_X, convert_W_to_wis(uniform_W), color="blue", marker="o")
rg = ax2.scatter(small_X, convert_W_to_wis(small_W), color="green", marker="o")
ax2.set_ylabel("Proportion")
ax2.set_ylim(0, 1)
ax.set_ylim(0, 1)
plt.title('D optimal design for Scenario Peaking 1 from Chapter 5')
legend_elements = [Line2D([0], [0], color='black', lw=4, label='Efficacy'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='r', label='D-Optimal', markersize=12),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='Uniform', markersize=12),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='g', label='3 near optimal', markersize=12)
                   ]
ax.legend(handles=legend_elements, loc='topleft')
plt.tight_layout()
plt.show()
