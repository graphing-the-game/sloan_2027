#find the sparse (defence, call) cells so our expert can give us a prior for them
#Marion Carr

import pandas as pd
from clean_data import DATA, OUTPUT, DEFS

#a cell is sparse if it has fewer corners than this
MIN_N = 20


if __name__ == '__main__':
    df = pd.read_excel(DATA / 'corners_cleaned.xlsx')

    calls = sorted(df['call'].unique())

    #corners and goals for each (defence, call)
    counts = pd.crosstab(df['def_group'], df['call']).reindex(index=DEFS, columns=calls, fill_value=0)
    goals = pd.crosstab(df['def_group'], df['call'], values=df['is_goal'], aggfunc='sum')
    goals = goals.reindex(index=DEFS, columns=calls).fillna(0).astype(int)

    print('corners per cell:')
    print(counts.to_string())
    print()

    #long format, one row per cell
    cells = pd.DataFrame({
        'defence': counts.stack().index.get_level_values(0),
        'call': counts.stack().index.get_level_values(1),
        'n': counts.stack().values,
        'goals': goals.stack().values,
    })
    cells['goal_rate'] = (cells['goals'] / cells['n']).round(3)

    sparse = cells[cells['n'] < MIN_N].sort_values('n').reset_index(drop=True)
    #blank column for the expert to fill in
    sparse['expert_prior_goal_rate'] = ''

    print(f'{len(sparse)} of {len(cells)} cells have fewer than {MIN_N} corners:')
    print(sparse.drop(columns='expert_prior_goal_rate').to_string(index=False))

    out = OUTPUT / 'sparse_cells_final_def.csv'
    sparse.to_csv(out, index=False)
    print(f'\nwrote {out.name}')
