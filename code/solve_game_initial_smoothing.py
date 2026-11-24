#solve the penalty corner game via linear programming
#Marion Carr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from clean_data import DATA, OUTPUT, DEFS

#laplace smoothing so sparse cells dont blow up, we report a table for each
ALPHAS = [0.2]

#short names for the table
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
        r"Defences are grouped by final formation: ``3-1'' pools ``3-1 to 3-1'' and ``2-2 to 3-1'' "
        r"(standard and reverse), ``2-2'' pools every set ending in 2-2; ``3 man defense'' is "
        r"excluded. ``Nash offense only'' applies the offensive Nash mix against the observed "
        r"defensive mix, and vice versa. Percentage changes are relative to observed goal rate.}",
        f"\\label{{tab:results_alpha{str(alpha).replace('.', '')}}}",
        r"\end{table}",
        "",
    ])


#plot how the Nash equilibrium changes as the Laplace smoothing parameter alpha changes
def make_alpha_sensitivity_plot(goals, counts, calls, alpha_grid=None, outpath=None):
    if alpha_grid is None:
        alpha_grid = np.linspace(0.01, 1.0, 100)

    off_mixes = []
    def_mixes = []
    values = []

    for alpha in alpha_grid:
        P = (goals + alpha) / (counts + 2 * alpha)
        off_mix, def_mix, value = solve_nash(P)
        off_mixes.append(off_mix)
        def_mixes.append(def_mix)
        values.append(value)

    off_mixes = np.asarray(off_mixes)
    def_mixes = np.asarray(def_mixes)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    for j, call in enumerate(calls):
        axes[0].plot(alpha_grid, 100 * off_mixes[:, j], label=SHORT.get(call, call))
    axes[0].axvline(0.2, linestyle='--', linewidth=1)
    axes[0].set_ylabel('Equilibrium probability (%)')
    axes[0].set_title('Offense equilibrium mix')
    axes[0].legend(fontsize=8, ncol=2)

    for j, defence in enumerate(DEFS):
        axes[1].plot(alpha_grid, 100 * def_mixes[:, j], label=defence)
    axes[1].axvline(0.2, linestyle='--', linewidth=1)
    axes[1].set_xlabel(r'Laplace smoothing parameter $\alpha$')
    axes[1].set_ylabel('Equilibrium probability (%)')
    axes[1].set_title('Defense equilibrium mix')
    axes[1].legend(fontsize=8, ncol=2)

    fig.suptitle('Nash equilibrium sensitivity to Laplace smoothing')
    fig.tight_layout()

    if outpath is None:
        outpath = OUTPUT / 'alpha_sensitivity.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return pd.DataFrame({
        'alpha': alpha_grid,
        'game_value': values,
    })


if __name__ == '__main__':
    df = pd.read_excel(DATA / 'corners_cleaned.xlsx')

    calls = sorted(df['call'].unique())

    #goals and corners for each (defence, call)
    goals = pd.crosstab(df['def_group'], df['call'], values=df['is_goal'], aggfunc='sum')
    goals = goals.reindex(index=DEFS, columns=calls).fillna(0).values
    counts = pd.crosstab(df['def_group'], df['call']).reindex(index=DEFS, columns=calls, fill_value=0).values

    #observed mixes
    obs_off = df['call'].value_counts(normalize=True).reindex(calls, fill_value=0).values
    obs_def = df['def_group'].value_counts(normalize=True).reindex(DEFS, fill_value=0).values

    tables = [make_table(goals, counts, calls, obs_off, obs_def, a) for a in ALPHAS]
    (OUTPUT / 'results_table.tex').write_text("\n".join(tables))

    #alpha robustness plot for the coarsened game used above
    make_alpha_sensitivity_plot(goals, counts, calls,
                                alpha_grid=np.linspace(0.01, 1.0, 100),
                                outpath=OUTPUT / 'alpha_sensitivity.png')
