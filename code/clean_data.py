#get corner data prepped
#Marion Carr

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
#raw and cleaned data live in data/, everything the analysis makes goes to output/
DATA = ROOT / 'data'
OUTPUT = ROOT / 'output'

#labels that arent real strategies
EXCLUDE = ['Error', 'Not observable', 'Other']

#coarsen the defences, high/low and standard/reverse variants get pooled
#defences not listed here ('Other', 'Not observable') get dropped
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

#order the defence groups show up in tables
DEFS = ['3-1 to 3-1', '3-1 to 2-2', '3-1 reverse', '2-2 to 3-1', '2-2 to 2-2',
        '3 man defense', 'Double runner']

#coarser grouping by final formation only
FINAL_GROUPS = {
    '3-1 to 3-1': '3-1',
    '2-2 to 3-1': '3-1',
    '2-2 to 3-1 reverse': '3-1',
    '3-1 to 3-1 reverse': '3-1 reverse',
    '3-1 to 2-2 low': '2-2',
    '3-1 to 2-2 high': '2-2',
    '2-2 to 2-2 low': '2-2',
    '2-2 to 2-2 high': '2-2',
    'Double runner': 'Double runner',
}
FINAL_DEFS = ['3-1', '3-1 reverse', '2-2', 'Double runner']


def load_and_clean():
    df = pd.read_excel(DATA / 'corner_data_anonymization_review.xlsx', sheet_name='Collaborator Data')
    n_raw = len(df)

    #drop missing and non strategic rows
    df = df.dropna(subset=['call', 'defence', 'is_goal'])
    df = df[~df['call'].isin(EXCLUDE)].copy()

    #coarsen defences and drop the ones that arent real formations
    df['def_group'] = df['defence'].map(DEF_GROUPS)
    df = df[df['def_group'].notna()]
    df['def_final'] = df['defence'].map(FINAL_GROUPS)

    print(f"kept {len(df)} of {n_raw} corners")
    return df


if __name__ == '__main__':
    df = load_and_clean()

    #save as xlsx so excel doesnt turn "3-1" into a date
    df.to_excel(DATA / 'corners_cleaned.xlsx', index=False)
    print(f"saved corners_cleaned.xlsx ({len(df)} rows)")
