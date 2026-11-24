#solve the penalty corner game via linear programming, sparse cells smoothed toward expert priors
#Marion Carr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

#the expert priors are on the final formation cells, so we use that grouping here
from clean_data import DATA, OUTPUT, FINAL_DEFS as DEFS

#k is how many corners the expert prior is worth, we report a table for each
KS = [10]

#ways to weight the experts, by their error on the check cells or by hand (manual)
WEIGHTINGS = ['equal', 'inv_err', 'inv_sq_err', 'manual']

#hand picked weights for the manual weighting, in survey order (expert 1 first)
#expert 1 has the most experience so he counts double
MANUAL_WEIGHTS = [2, 1, 1, 1, 1]

#survey labels that are written differently in the data
SURVEY_DEFS = {'Reversed 3-1': '3-1 reverse'}
SURVEY_CALLS = {'Back to inserter': 'Back to Inserter'}

#well observed cells on the survey, used to check the experts and not as priors
CHECK_CELLS = [('3-1', 'Pass or slip to stick side'), ('3-1', 'Back to Inserter')]

#a cell is sparse if it has fewer corners than this (same as find_sparce_cells.py)
MIN_N = 20

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


#read the survey, each (defence, call) cell maps to every experts answer as a probability
def load_survey(calls):
    survey = pd.read_excel(DATA / 'penalty_corner_survey.xlsx', sheet_name='Survey responses')
    answers = {}
    for col in survey.columns[1:]:
        defence, call = [x.strip() for x in col.split('—')]
        defence = SURVEY_DEFS.get(defence, defence)
        call = SURVEY_CALLS.get(call, call)
        answers[(DEFS.index(defence), calls.index(call))] = survey[col].values / 100
    return answers


#weight each expert by how close they got on the check cells
def expert_weights(goals, counts, answers, calls, weighting):
    checks = [(DEFS.index(d), calls.index(c)) for d, c in CHECK_CELLS]
    #rows are check cells, columns are experts
    err = np.array([answers[ij] - goals[ij] / counts[ij] for ij in checks])

    if weighting == 'equal':
        w = np.ones(err.shape[1])
    elif weighting == 'inv_err':
        w = 1 / np.abs(err).mean(axis=0)
    elif weighting == 'inv_sq_err':
        w = 1 / (err ** 2).mean(axis=0)
    elif weighting == 'manual':
        w = np.array(MANUAL_WEIGHTS, dtype=float)
    return w / w.sum()


#prior mean for every cell: weighted expert answer where we asked, overall goal rate everywhere else
def prior_means(goals, counts, answers, calls, weighting):
    m = np.full(goals.shape, goals.sum() / counts.sum())
    if weighting is None:
        return m

    w = expert_weights(goals, counts, answers, calls, weighting)
    checks = [(DEFS.index(d), calls.index(c)) for d, c in CHECK_CELLS]
    for ij, a in answers.items():
        if ij not in checks:
            m[ij] = w @ a
    return m


#posterior mean of a beta(k m, k (1 - m)) prior after g goals in n corners
def posterior_P(goals, counts, m, k):
    return (goals + k * m) / (counts + k)


#one table cell listing a mix, biggest first
def mix_cell(labels, mix):
    lines = [f"{SHORT.get(labels[i], labels[i])} {mix[i] * 100:.1f}\\%"
             for i in np.argsort(-mix) if mix[i] > 0.0005]
    return r"\begin{tabular}[t]{@{}l@{}} " + r" \\ ".join(lines) + r" \end{tabular}"


#solve the game for one k and weighting and build its latex table
def make_table(goals, counts, m, calls, obs_off, obs_def, k, weighting):
    P = posterior_P(goals, counts, m, k)
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
        r"\caption{Nash equilibrium results. Payoff matrix entries are posterior means under a "
        f"Beta prior worth $k = {k}$ corners, centred on the expert estimate "
        f"(experts weighted: {weighting.replace('_', ' ')}) for surveyed sparse cells and on the "
        r"overall goal rate elsewhere. Defences are grouped by final formation: ``3-1'' pools "
        r"``3-1 to 3-1'' and ``2-2 to 3-1'' (standard and reverse), ``2-2'' pools every set ending "
        r"in 2-2; ``3 man defense'' was not surveyed and is excluded. ``Nash offense only'' applies "
        r"the offensive Nash mix against the observed defensive mix, and vice versa. Percentage "
        r"changes are relative to observed goal rate.}",
        f"\\label{{tab:results_bayes_{weighting}_k{k}}}",
        r"\end{table}",
        "",
    ])


#plot how the Nash equilibrium changes as the prior strength k changes
def make_k_sensitivity_plot(goals, counts, m, calls, weighting, k_grid=None, outpath=None):
    if k_grid is None:
        k_grid = np.geomspace(0.5, 100, 100)

    off_mixes = []
    def_mixes = []
    values = []

    for k in k_grid:
        P = posterior_P(goals, counts, m, k)
        off_mix, def_mix, value = solve_nash(P)
        off_mixes.append(off_mix)
        def_mixes.append(def_mix)
        values.append(value)

    off_mixes = np.asarray(off_mixes)
    def_mixes = np.asarray(def_mixes)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    for j, call in enumerate(calls):
        axes[0].plot(k_grid, 100 * off_mixes[:, j], label=SHORT.get(call, call))
    axes[0].axvline(KS[0], linestyle='--', linewidth=1)
    axes[0].set_ylabel('Equilibrium probability (%)')
    axes[0].set_title('Offense equilibrium mix')
    axes[0].legend(fontsize=8, ncol=2)

    for j, defence in enumerate(DEFS):
        axes[1].plot(k_grid, 100 * def_mixes[:, j], label=defence)
    axes[1].axvline(KS[0], linestyle='--', linewidth=1)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Prior strength $k$ (corners)')
    axes[1].set_ylabel('Equilibrium probability (%)')
    axes[1].set_title('Defense equilibrium mix')
    axes[1].legend(fontsize=8, ncol=2)

    fig.suptitle(f"Nash equilibrium sensitivity to expert prior strength ({weighting.replace('_', ' ')} weights)")
    fig.tight_layout()

    if outpath is None:
        outpath = OUTPUT / f'k_sensitivity_{weighting}.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return pd.DataFrame({
        'k': k_grid,
        'game_value': values,
    })


#held out log loss for each corner, folds are by game so a games corners stay together
#make_P builds the payoff matrix from the training goals and counts only
def cv_log_loss(df, calls, make_P, n_folds=10, seed=0):
    d = df['def_final'].map({x: i for i, x in enumerate(DEFS)}).values
    c = df['call'].map({x: j for j, x in enumerate(calls)}).values
    y = df['is_goal'].values.astype(float)

    games = np.unique(df['game_id'])
    fold_of = dict(zip(games, np.random.default_rng(seed).permutation(len(games)) % n_folds))
    fold = df['game_id'].map(fold_of).values

    loss = np.zeros(len(df))
    for f in range(n_folds):
        train, test = fold != f, fold == f
        goals = np.zeros((len(DEFS), len(calls)))
        counts = np.zeros_like(goals)
        np.add.at(goals, (d[train], c[train]), y[train])
        np.add.at(counts, (d[train], c[train]), 1)

        p = np.clip(make_P(goals, counts), 1e-6, 1 - 1e-6)[d[test], c[test]]
        loss[test] = -(y[test] * np.log(p) + (1 - y[test]) * np.log(1 - p))
    return loss


#compare every approach out of sample against the laplace prior in the paper
def compare_out_of_sample(df, calls, counts, answers, k_grid=(2, 5, 10, 20, 50, 100)):
    methods = {'laplace alpha=0.2': lambda g, n: (g + 0.2) / (n + 0.4)}
    #no experts: same shrinkage but toward the overall goal rate, tells us if the experts add anything
    for k in k_grid:
        methods[f'no experts k={k}'] = (
            lambda k: lambda g, n: posterior_P(g, n, prior_means(g, n, answers, calls, None), k))(k)
    for weighting in WEIGHTINGS:
        for k in k_grid:
            methods[f'{weighting} k={k}'] = (
                lambda k, w: lambda g, n: posterior_P(g, n, prior_means(g, n, answers, calls, w), k))(k, weighting)

    d = df['def_final'].map({x: i for i, x in enumerate(DEFS)}).values
    c = df['call'].map({x: j for j, x in enumerate(calls)}).values
    sparse = counts[d, c] < MIN_N

    losses = {name: cv_log_loss(df, calls, make_P) for name, make_P in methods.items()}
    base = losses['laplace alpha=0.2']

    rows = []
    for name, loss in losses.items():
        diff = loss - base
        rows.append({
            'method': name,
            'log_loss': loss.mean(),
            'vs_laplace': diff.mean(),
            'se': diff.std() / np.sqrt(len(diff)),
            'log_loss_sparse': loss[sparse].mean(),
            'vs_laplace_sparse': diff[sparse].mean(),
            'se_sparse': diff[sparse].std() / np.sqrt(sparse.sum()),
        })
    results = pd.DataFrame(rows)

    print(f'\nheld out log loss, 10 folds by game ({len(df)} corners, {sparse.sum()} in sparse cells)')
    print('vs_laplace < 0 means better than the laplace prior in the paper')
    print(results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return results


if __name__ == '__main__':
    df = pd.read_excel(DATA / 'corners_cleaned.xlsx')

    #3 man defense wasnt on the survey so it has no final formation group
    df = df[df['def_final'].notna()]
    calls = sorted(df['call'].unique())

    #goals and corners for each (defence, call)
    goals = pd.crosstab(df['def_final'], df['call'], values=df['is_goal'], aggfunc='sum')
    goals = goals.reindex(index=DEFS, columns=calls).fillna(0).values
    counts = pd.crosstab(df['def_final'], df['call']).reindex(index=DEFS, columns=calls, fill_value=0).values

    #observed mixes
    obs_off = df['call'].value_counts(normalize=True).reindex(calls, fill_value=0).values
    obs_def = df['def_final'].value_counts(normalize=True).reindex(DEFS, fill_value=0).values

    answers = load_survey(calls)

    tables = []
    for weighting in WEIGHTINGS:
        print(f"{weighting} expert weights: {np.round(expert_weights(goals, counts, answers, calls, weighting), 3)}")
        m = prior_means(goals, counts, answers, calls, weighting)
        tables += [make_table(goals, counts, m, calls, obs_off, obs_def, k, weighting) for k in KS]

        #k robustness plot for this weighting
        make_k_sensitivity_plot(goals, counts, m, calls, weighting,
                                k_grid=np.geomspace(0.5, 100, 100),
                                outpath=OUTPUT / f'k_sensitivity_{weighting}.png')

    (OUTPUT / 'results_table_bayes.tex').write_text("\n".join(tables))

    compare_out_of_sample(df, calls, counts, answers)
