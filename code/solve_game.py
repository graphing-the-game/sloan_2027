#solve the penalty corner game via linear programming
#Marion Carr

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]

#labels that arent real strategies
EXCLUDE = ['Error', 'Not observable', 'Other']

#coarsen the defences, high/low and standard/reverse variants get pooled
DEF_GROUPS = {
    '3-1 to 3-1': '3-1 to 3-1',
    '3-1 to 2-2 low': '3-1 to 2-2',
    '3-1 to 2-2 high': '3-1 to 2-2',
    '3-1 to 3-1 reverse': '3-1 reverse',
    '2-2 to 3-1': '2-2 to 3-1',
    '2-2 to 3-1 reverse': '2-2 to 3-1',
    '2-2 to 2-2 low': '2-2 to 2-2',
    '2-2 to 2-2 high': '2-2 to 2-2',
    '3 man defense': '3 man defense',
    'Double runner': 'Double runner',
}
DEFS = ['3-1 to 3-1', '3-1 to 2-2', '3-1 reverse', '2-2 to 3-1', '2-2 to 2-2',
        '3 man defense', 'Double runner']

#laplace smoothing so sparse cells dont blow up, we report a table for each
ALPHAS = [0.2]

#short names for the table, same as the note
SHORT = {
    'Pass or slip to stick side': 'Pass/stick',
    'Pass or slip to glove side': 'Pass/glove',
    'Back to Inserter': 'BTI',
    'Opposite side insert': 'OSI',
}


#solve the game, rows are defense (minimizes) and columns are offense (maximizes)
def solve_nash(P):
    m, n = P.shape

    #offense: max v s.t. P @ p >= v, sum(p) = 1, p >= 0
    c = np.zeros(n + 1)
    c[-1] = -1
    A_eq = np.append(np.ones(n), 0).reshape(1, -1)
    res_off = linprog(c, A_ub=np.hstack([-P, np.ones((m, 1))]), b_ub=np.zeros(m),
                      A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * n + [(None, None)], method='highs')

    #defense: min w s.t. P^T @ q <= w, sum(q) = 1, q >= 0
    c = np.zeros(m + 1)
    c[-1] = 1
    A_eq = np.append(np.ones(m), 0).reshape(1, -1)
    res_def = linprog(c, A_ub=np.hstack([P.T, -np.ones((n, 1))]), b_ub=np.zeros(n),
                      A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * m + [(None, None)], method='highs')

    if not (res_off.success and res_def.success):
        raise RuntimeError('lp failed')

    return res_off.x[:n], res_def.x[:m], -res_off.fun


#one table cell listing a mix, biggest first
def mix_cell(labels, mix):
    lines = [f"{SHORT.get(labels[i], labels[i])} {mix[i] * 100:.1f}\\%"
             for i in np.argsort(-mix) if mix[i] > 0.0005]
    return r"\begin{tabular}[t]{@{}l@{}} " + r" \\ ".join(lines) + r" \end{tabular}"


#solve the game for one alpha and build its latex table
def make_table(goals, counts, calls, obs_off, obs_def, alpha):
    P = (goals + alpha) / (counts + 2 * alpha)
    off_mix, def_mix, value = solve_nash(P)

    #goal rate under each scenario
    rates = [
        obs_def @ P @ obs_off,  # observed
        obs_def @ P @ off_mix,  # nash offense only
        def_mix @ P @ obs_off,  # nash defense only
        value,                  # nash equilibrium
    ]

    same = r"\begin{tabular}[t]{@{}l@{}} (unchanged) \end{tabular}"
    goal_cells = [f"{rates[0] * 100:.1f}\\%"] + [
        f"{r * 100:.1f}\\% (${(r / rates[0] - 1) * 100:+.1f}\\%$)" for r in rates[1:]]

    return "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r" & Observed & Nash offense only & Nash defense only & Nash equilibrium \\",
        r"\midrule",
        r"Offense $\sigma^{O}$ &",
        mix_cell(calls, obs_off) + " &",
        mix_cell(calls, off_mix) + " &",
        same + " &",
        mix_cell(calls, off_mix) + r" \\",
        r"\midrule",
        r"Defense $\sigma^{D}$ &",
        mix_cell(DEFS, obs_def) + " &",
        same + " &",
        mix_cell(DEFS, def_mix) + " &",
        mix_cell(DEFS, def_mix) + r" \\",
        r"\midrule",
        "Goal Rate & " + " & ".join(goal_cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Nash equilibrium results. Payoff matrix entries smoothed via Laplace prior "
        f"($\\alpha = {alpha}$). "
        r"Defence categories are coarsened: ``3-1 to 2-2'' and ``2-2 to 2-2'' pool low and high "
        r"variants; ``2-2 to 3-1'' pools standard and reverse. ``Nash offense only'' applies the "
        r"offensive Nash mix against the observed defensive mix, and vice versa. Percentage "
        r"changes are relative to observed goal rate.}",
        f"\\label{{tab:results_alpha{str(alpha).replace('.', '')}}}",
        r"\end{table}",
        "",
    ])


if __name__ == '__main__':
    df = pd.read_excel(ROOT / 'corners_cleaned.xlsx')

    #drop missing and non strategic rows, then coarsen defences
    sub = df.dropna(subset=['call', 'defence', 'is_goal'])
    sub = sub[~sub['call'].isin(EXCLUDE)].copy()
    sub['def_c'] = sub['defence'].map(DEF_GROUPS)
    sub = sub[sub['def_c'].notna()]
    calls = sorted(sub['call'].unique())

    #goals and corners for each (defence, call)
    goals = pd.crosstab(sub['def_c'], sub['call'], values=sub['is_goal'], aggfunc='sum')
    goals = goals.reindex(index=DEFS, columns=calls).fillna(0).values
    counts = pd.crosstab(sub['def_c'], sub['call']).reindex(index=DEFS, columns=calls, fill_value=0).values

    #observed mixes
    obs_off = sub['call'].value_counts(normalize=True).reindex(calls, fill_value=0).values
    obs_def = sub['def_c'].value_counts(normalize=True).reindex(DEFS, fill_value=0).values

    tables = [make_table(goals, counts, calls, obs_off, obs_def, a) for a in ALPHAS]
    (ROOT / 'results_table.tex').write_text("\n".join(tables))
