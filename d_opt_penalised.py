import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from particle import Swarm


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


def check_symmetric(a, tol=1e-6):
    return np.all(np.abs(a - a.T) < tol)


def efficiency(detM, detM_opt, k):
    return (detM / detM_opt) ** (1 / k)


def get_desire2(X, W):
    wis = convert_W_to_wis(W)
    return desirability_of_design(X, wis, eff,
                                  m=2.0, efficacy_min=worst, efficacy_max=best)


def get_desire5(X, W):
    wis = convert_W_to_wis(W)
    return desirability_of_design(X, wis, eff,
                                  m=0.5, efficacy_min=worst, efficacy_max=best)


def get_desire8(X, W):
    wis = convert_W_to_wis(W)
    return desirability_of_design(X, wis, eff,
                                  m=8.0, efficacy_min=worst, efficacy_max=best)


def get_ef(X, W):
    wis = convert_W_to_wis(W)
    return ef_of_design(X, wis, eff,
                        m=8.0, efficacy_min=worst, efficacy_max=best)


a, b, c = -9.0, 3.0, -0.214
query_doses = np.linspace(0, 10, 101)
response = h(z(query_doses, a, b, c))
worst, best = np.min(response), np.max(response)
lam = 1
m = 2

uniform_X = np.linspace(0, 10, 31)
uniform_W = np.ones(30) / 31
small_X = np.linspace(6.5, 7.5, 3)
small_W = np.ones(2) / 3


def eff(X):
    return h(z(X, a, b, c))


def desirability_i(efficacy_i, m=1, efficacy_min=0, efficacy_max=1):
    return ((efficacy_i - efficacy_min) / (efficacy_max - efficacy_min)) ** m


def desirability_of_design(X, wis, efficacy_function, m=1.0, efficacy_min=0, efficacy_max=1):
    _E = efficacy_function(X)
    E = _E * wis
    efficacy = np.sum(E)
    return desirability_i(efficacy, m, efficacy_min, efficacy_max)


def fitness(X, W):
    wis = convert_W_to_wis(W)
    _Mw = Mw(X, wis, a, b, c)
    return detM(_Mw)


def ef_of_design(X, wis, efficacy_function, m=1.0, efficacy_min=0, efficacy_max=1):
    _E = efficacy_function(X)
    E = _E * wis
    efficacy = np.sum(E)
    return efficacy


def penalised_fitness2(X, W):
    wis = convert_W_to_wis(W)
    _Mw = Mw(X, wis, a, b, c)
    return detM(_Mw) * desirability_of_design(X, wis, eff,
                                              m=2.0, efficacy_min=worst, efficacy_max=best) ** lam


def penalised_fitness5(X, W):
    wis = convert_W_to_wis(W)
    _Mw = Mw(X, wis, a, b, c)
    return detM(_Mw) * desirability_of_design(X, wis, eff,
                                              m=0.5, efficacy_min=worst, efficacy_max=best) ** lam


def penalised_fitness8(X, W):
    wis = convert_W_to_wis(W)
    _Mw = Mw(X, wis, a, b, c)
    return detM(_Mw) * desirability_of_design(X, wis, eff,
                                              m=8.0, efficacy_min=worst, efficacy_max=best) ** lam


plt.plot(query_doses, response, c='black')
m_array = [2.0, 8.0, 32.0]
c_array = ['gray', 'purple', 'brown']
legend_elements = [Line2D([0], [0], color='black', lw=4, label='Efficacy')
                   ]
for _m, _c in zip(m_array, c_array):
    d = desirability_i(response, _m, np.min(response), np.max(response))
    plt.plot(query_doses, d, color=_c)
    legend_elements.append(Line2D([0], [0], color=_c, lw=4, label='Desirability for m = ' + str(_m)))
plt.legend(handles=legend_elements, loc='topleft')
plt.ylabel('Desirability of design with all trial individuals at dose')
plt.xlabel('Dose')
plt.title('Effect of m parameter on desirability')
plt.tight_layout()
plt.show()
plt.close()

l = np.linspace(np.min(response), np.max(response), 101)
for _m, _c in zip(m_array, c_array):
    d = desirability_i(l, _m, np.min(response), np.max(response))
    plt.plot(l, d, color=_c)
plt.legend(handles=legend_elements, loc='topleft')
plt.ylabel('Desirability of design with mean efficacy for m')
plt.xlabel('Mean Efficacy')
plt.xlim(0, 1)
plt.title('Effect of m parameter on desirability')
plt.tight_layout()
plt.show()
plt.close()

s = Swarm(20, 3, 10.0, 0.0, fitness)
s.optimise(100)
sp2 = Swarm(20, 3, 10.0, 0.0, penalised_fitness2)
sp2.optimise(100)
sp5 = Swarm(20, 3, 10.0, 0.0, penalised_fitness5)
sp5.optimise(100)
sp8 = Swarm(20, 3, 10.0, 0.0, penalised_fitness8)
sp8.optimise(100)

df = pd.DataFrame({
    'Design Name': [],
    'Efficiency': [],
    'Score m = 2.0': [],
    'Score m = 8.0': [],
    'Score m = 32.0': [],
    'Desirability m = 2.0': [],
    'Desirability m = 8.0': [],
    'Desirability m = 32.0': [],
    'Mean Efficacy': []
})
names = ['D-optimal', 'Penalised m = 2.0', 'Penalised m = 8.0', 'Penalised m = 32.0', 'Uniform 30',
         'Three near optimal']
Xs = [s.X_best, sp5.X_best, sp2.X_best, sp8.X_best, uniform_X, small_X]
Ws = [s.W_best, sp5.W_best, sp2.W_best, sp8.W_best, uniform_W, small_W]
for n, x, w in zip(names, Xs, Ws):
    df2 = pd.DataFrame({
        'Design Name': [n],
        'Efficiency': [efficiency(fitness(x, w), s.global_fitness_best, k=3)],
        'Score m = 2.0': [penalised_fitness5(x, w)],
        'Score m = 8.0': [penalised_fitness2(x, w)],
        'Score m = 32.0': [penalised_fitness8(x, w)],
        'Desirability m = 2.0': [get_desire5(x, w)],
        'Desirability m = 8.0': [get_desire2(x, w)],
        'Desirability m = 32.0': [get_desire8(x, w)],
        'Mean Efficacy': [get_ef(x, w)]
    })
    df = df.append(df2)
df.to_csv('penalised_table.csv', float_format='%.2f')

print(f'For unpenalised: Efficiency D: {1.0},'
      f' Penalised Score m = 0.5 : {penalised_fitness5(s.X_best, s.W_best)} '
      f'Penalised Score m = 2.0 : {penalised_fitness2(s.X_best, s.W_best)} '
      f'Penalised Score m = 8.0 : {penalised_fitness8(s.X_best, s.W_best)} ,'
      f' desire m = 0.5: {get_desire5(s.X_best, s.W_best)},'
      f'desire m = 2: {get_desire2(s.X_best, s.W_best)},'
      f'desire m = 8: {get_desire8(s.X_best, s.W_best)}')
print(f'For pen5: Efficiency D: {efficiency(fitness(sp5.X_best, sp5.W_best), s.global_fitness_best, k=3)},'
      f' Penalised Score m = 0.5 : {penalised_fitness5(sp5.X_best, sp5.W_best)} '
      f'Penalised Score m = 2.0 : {penalised_fitness2(sp5.X_best, sp5.W_best)} '
      f'Penalised Score m = 8.0 : {penalised_fitness8(sp5.X_best, sp5.W_best)} ,'
      f' desire m = 0.5: {get_desire5(sp5.X_best, sp5.W_best)},'
      f'desire m = 2: {get_desire2(sp5.X_best, sp5.W_best)},'
      f'desire m = 8: {get_desire8(sp5.X_best, sp5.W_best)}')
print(f'For pen2: Efficiency D: {efficiency(fitness(sp2.X_best, sp2.W_best), s.global_fitness_best, k=3)},'
      f' Penalised Score m = 0.5 : {penalised_fitness5(sp2.X_best, sp2.W_best)} '
      f'Penalised Score m = 2.0 : {penalised_fitness2(sp2.X_best, sp2.W_best)} '
      f'Penalised Score m = 8.0 : {penalised_fitness8(sp2.X_best, sp2.W_best)} ,'
      f' desire m = 0.5: {get_desire5(sp2.X_best, sp2.W_best)},'
      f'desire m = 2: {get_desire2(sp2.X_best, sp2.W_best)},'
      f'desire m = 8: {get_desire8(sp2.X_best, sp2.W_best)}')
print(f'For pen8: Efficiency D: {efficiency(fitness(sp8.X_best, sp8.W_best), s.global_fitness_best, k=3)},'
      f' Penalised Score m = 0.5 : {penalised_fitness5(sp8.X_best, sp8.W_best)} '
      f'Penalised Score m = 2.0 : {penalised_fitness2(sp8.X_best, sp8.W_best)} '
      f'Penalised Score m = 8.0 : {penalised_fitness8(sp8.X_best, sp8.W_best)} ,'
      f' desire m = 0.5: {get_desire5(sp8.X_best, sp8.W_best)},'
      f'desire m = 2: {get_desire2(sp8.X_best, sp8.W_best)},'
      f'desire m = 8: {get_desire8(sp8.X_best, sp8.W_best)}')
print(f'For uniform: Efficiency D: {efficiency(fitness(uniform_X, uniform_W), s.global_fitness_best, k=3)},'
      f' Penalised Score m = 0.5 : {penalised_fitness5(uniform_X, uniform_W)} '
      f'Penalised Score m = 2.0 : {penalised_fitness2(uniform_X, uniform_W)} '
      f'Penalised Score m = 8.0 : {penalised_fitness8(uniform_X, uniform_W)} ,'
      f' desire m = 0.5: {get_desire5(uniform_X, uniform_W)},'
      f'desire m = 2: {get_desire2(uniform_X, uniform_W)},'
      f'desire m = 8: {get_desire8(uniform_X, uniform_W)}')
print(f'For 3 near optimal: Efficiency D: {efficiency(fitness(small_X, small_W), s.global_fitness_best, k=3)},'
      f' Penalised Score m = 0.5 : {penalised_fitness5(small_X, small_W)} '
      f'Penalised Score m = 2.0 : {penalised_fitness2(small_X, small_W)} '
      f'Penalised Score m = 8.0 : {penalised_fitness8(small_X, small_W)} ,'
      f' desire m = 0.5: {get_desire5(small_X, small_W)},'
      f'desire m = 2: {get_desire2(small_X, small_W)},'
      f'desire m = 8: {get_desire8(small_X, small_W)}')

# create figure and axis objects with subplots()
fig, ax = plt.subplots()
ax.plot(query_doses, response, color='black')
ax.set_xlabel("Dose")
ax.set_ylabel("Efficacy probability")

ax2 = ax.twinx()
ax2.scatter(sp5.X_best, convert_W_to_wis(sp5.W_best), color="gray", marker="o")
ax2.scatter(sp2.X_best, convert_W_to_wis(sp2.W_best), color="purple", marker="o")
ax2.scatter(sp8.X_best, convert_W_to_wis(sp8.W_best), color="brown", marker="o")
ax2.scatter(s.X_best, convert_W_to_wis(s.W_best), color="red", marker="o")
ax2.scatter(uniform_X, convert_W_to_wis(uniform_W), color="blue", marker="o")
ax2.scatter(small_X, convert_W_to_wis(small_W), color="green", marker="o")
ax2.set_ylabel("Proportion")
ax2.set_ylim(0, 1)
ax.set_ylim(0, 1)
plt.title('D optimal design for Scenario Peaking 1 from Chapter 5')

legend_elements = [Line2D([0], [0], color='black', lw=4, label='Efficacy'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='r', label='D-Optimal', markersize=12),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='Uniform', markersize=12),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='g', label='3 near optimal', markersize=12),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Penalised D-Optimal m = 0.5',
                          markersize=12),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
                          label='Penalised D-Optimal m = 2.0', markersize=12),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', label='Penalised D-Optimal m = 8.0',
                          markersize=12)
                   ]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()
