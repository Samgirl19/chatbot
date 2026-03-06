import pandas as pd

def load_data():
    df = pd.read_csv("biscayneBay_waterquality.csv")
    return df

def summarize_data(df):

    summary = f"""
Rows: {len(df)}

Average pH: {df['pH'].mean():.2f}
Average Turbidity: {df['turbidity'].mean():.2f}
Average Dissolved Oxygen: {df['dissolved_oxygen'].mean():.2f}

Locations:
{', '.join(df['location'].tolist())}
"""

    return summary