#toy example
#Marion Carr

import numpy as np
from scipy.optimize import linprog

#lets do a 2x2 and make sure this works
#create payoff matrix where we will have a mixed nash
#rows are defense, columns are offense, entries are P(goal)
U = np.array([
    [0.25, 0.75],
    [0.75, 0.25]
])


#create payoffs
def offense_expected_payoff(p, q):
    # E[u] = p*q*U[0,0] + p*(1-q)*U[1,0] + (1-p)*q*U[0,1] + (1-p)*(1-q)*U[1,1]
    off_mix = np.array([p, 1 - p])
    def_mix = np.array([q, 1 - q])
    return def_mix @ U @ off_mix


#offense best response to the defense playing A1_D with prob q
def offense_best_response(q):
    # E[u | A1_O] = q*U[0,0] + (1-q)*U[1,0]
    # E[u | A2_O] = q*U[0,1] + (1-q)*U[1,1]
    eu_a1 = q * U[0, 0] + (1 - q) * U[1, 0]
    eu_a2 = q * U[0, 1] + (1 - q) * U[1, 1]

    if eu_a1 > eu_a2:
        return 1.0
    elif eu_a2 > eu_a1:
        return 0.0
    else:
        return 0.5  # indifferent so any p is a br


def defense_best_response(p):
    #defense best response to the offense playing A1_O with prob p, defense minimizes
    # E[u | A1_D] = p*U[0,0] + (1-p)*U[0,1]
    # E[u | A2_D] = p*U[1,0] + (1-p)*U[1,1]
    eu_a1 = p * U[0, 0] + (1 - p) * U[0, 1]
    eu_a2 = p * U[1, 0] + (1 - p) * U[1, 1]

    if eu_a1 < eu_a2:
        return 1.0
    elif eu_a2 < eu_a1:
        return 0.0
    else:
        return 0.5  # indifferent


#find nash with indifference conditions
#offense indifference: q*U[0,0] + (1-q)*U[1,0] = q*U[0,1] + (1-q)*U[1,1]
#solve for q
q_star = (U[1, 1] - U[1, 0]) / (U[0, 0] - U[0, 1] - U[1, 0] + U[1, 1])

#defense indifference: p*U[0,0] + (1-p)*U[0,1] = p*U[1,0] + (1-p)*U[1,1]
#solve for p
p_star = (U[1, 1] - U[0, 1]) / (U[0, 0] - U[1, 0] - U[0, 1] + U[1, 1])

game_value = offense_expected_payoff(p_star, q_star)

print("payoff matrix (P(goal)):")
print(f"         A1_O    A2_O")
print(f"  A1_D   {U[0,0]:.2f}    {U[0,1]:.2f}")
print(f"  A2_D   {U[1,0]:.2f}    {U[1,1]:.2f}")
print()
print(f"nash equilibrium: p* = {p_star:.4f}, q* = {q_star:.4f}")
print(f"game value: {game_value:.4f}")
print()

#check the best responses
print("verification:")
print(f"  Offense BR(q*={q_star:.2f}) = {offense_best_response(q_star)}")
print(f"  Defense BR(p*={p_star:.2f}) = {defense_best_response(p_star)}")
print()


#now do it as a linear program, this is what solve_game.py uses for the big matrix
m, n = U.shape  # m defense actions, n offense actions

#offense lp
c_off = np.zeros(n + 1)
c_off[-1] = -1  # min -v

A_ub_off = np.hstack([-U, np.ones((m, 1))])  # -U @ p + v <= 0
b_ub_off = np.zeros(m)

A_eq_off = np.zeros((1, n + 1))
A_eq_off[0, :n] = 1  # sum(p) = 1
b_eq_off = np.array([1.0])

bounds_off = [(0, None)] * n + [(None, None)]  # p >= 0, v unrestricted

res_off = linprog(c_off, A_ub=A_ub_off, b_ub=b_ub_off,
                  A_eq=A_eq_off, b_eq=b_eq_off, bounds=bounds_off, method='highs')

p_lp = res_off.x[:n]
v_off = -res_off.fun

#defense lp (minimizes): min w s.t. U^T @ q <= w, sum(q) = 1, q >= 0
c_def = np.zeros(m + 1)
c_def[-1] = 1  # min w

A_ub_def = np.hstack([U.T, -np.ones((n, 1))])  # U^T @ q - w <= 0
b_ub_def = np.zeros(n)

A_eq_def = np.zeros((1, m + 1))
A_eq_def[0, :m] = 1
b_eq_def = np.array([1.0])

bounds_def = [(0, None)] * m + [(None, None)]

res_def = linprog(c_def, A_ub=A_ub_def, b_ub=b_ub_def,
                  A_eq=A_eq_def, b_eq=b_eq_def, bounds=bounds_def, method='highs')

q_lp = res_def.x[:m]
v_def = res_def.fun

print("=" * 50)
print("method 1: indifference conditions (2x2 only)")
print(f"  p* = {p_star:.4f}, q* = {q_star:.4f}, value = {game_value:.4f}")
print()
print("method 2: linear program (works for any n x m)")
print(f"  p  = [{p_lp[0]:.4f}, {p_lp[1]:.4f}]")
print(f"  q  = [{q_lp[0]:.4f}, {q_lp[1]:.4f}]")
print(f"  offense value = {v_off:.4f}")
print(f"  defense value = {v_def:.4f}")
print(f"  (these should match: {np.isclose(v_off, v_def)})")
print("=" * 50)
print()
