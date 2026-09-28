#get corner data prepped
#Marion Carr

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_and_clean():
    df = pd.read_excel(ROOT / 'corner_data_anonymization_review.xlsx', sheet_name='Collaborator Data')

    #split defence at ' to ' into initial and final formation
    #if theres no ' to ' just keep the whole thing as the final one
    parts = df['defence'].str.split(' to ', n=1, expand=True)
    has_to = parts[1].notna()
    df['defence_initial'] = parts[0].where(has_to)
    df['defence_final'] = parts[1].where(has_to, df['defence'])

    return df


if __name__ == '__main__':
    df = load_and_clean()

    #save as xlsx so excel doesnt turn "3-1" into a date
    df.to_excel(ROOT / 'corners_cleaned.xlsx', index=False)
    print(f"saved corners_cleaned.xlsx ({len(df)} rows)")
