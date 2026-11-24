#solve the penalty corner game via linear programming, sparse cells smoothed toward expert priors
#Marion Carr

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from clean_data import DATA, OUTPUT, DEFS

#k is how many corners the expert prior is worth
#k = 20 matches the sparse cutoff, so a cell with 20 corners weights the data and the experts equally
K = 20

#survey labels that are written differently in the data
SURVEY_DEFS = {'Reversed 3-1': '3-1 reverse'}
SURVEY_CALLS = {'Back to inserter': 'Back to Inserter'}

#well observed cells on the survey, used to check the experts and not as priors
CHECK_CELLS = [('3-1', 'Pass or slip to stick side'), ('3-1', 'Back to Inserter')]

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


#weight each expert by how close they got on the check cells (inverse mean absolute error)
def expert_weights(goals, counts, answers, calls):
    checks = [(DEFS.index(d), calls.index(c)) for d, c in CHECK_CELLS]
    #rows are check cells, columns are experts
    err = np.array([answers[ij] - goals[ij] / counts[ij] for ij in checks])
    w = 1 / np.abs(err).mean(axis=0)
    return w / w.sum()


#prior mean for every cell: weighted expert answer where we asked, overall goal rate everywhere else
def prior_means(goals, counts, answers, calls):
    m = np.full(goals.shape, goals.sum() / counts.sum())
    w = expert_weights(goals, counts, answers, calls)
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


#solve the game for one k and build its latex table
def make_table(goals, counts, m, calls, obs_off, obs_def, k):
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
        r"(experts weighted by inverse mean error on two check questions) for surveyed sparse "
        r"cells and on the overall goal rate elsewhere. Defences are grouped by final formation: ``3-1'' pools "
        r"``3-1 to 3-1'' and ``2-2 to 3-1'' (standard and reverse), ``2-2'' pools every set ending "
        r"in 2-2; ``3 man defense'' is excluded. ``Nash offense only'' applies "
        r"the offensive Nash mix against the observed defensive mix, and vice versa. Percentage "
        r"changes are relative to observed goal rate.}",
        f"\\label{{tab:results_bayes_k{k}}}",
        r"\end{table}",
        "",
    ])


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

    answers = load_survey(calls)
    print(f"expert weights: {np.round(expert_weights(goals, counts, answers, calls), 3)}")

    m = prior_means(goals, counts, answers, calls)
    (OUTPUT / 'results_table_bayes.tex').write_text(make_table(goals, counts, m, calls, obs_off, obs_def, K))
