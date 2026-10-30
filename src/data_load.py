
import pandas as pd


def Issue_col(df):
    drop_these = df[df.duplicated(subset='Issue')].sample(frac=0.72, random_state=42).index
    df = df.drop(drop_these)

    df['Issue'] = df["Issue"].str.lower().str.strip()
    df=df[df['Issue'].str.len()>15]
    return df

def Product_col(df):
    df['Product'] = df['Product'].replace({'Credit reporting, credit repair services, or other personal consumer reports': 'Credi and other personal'})

    top=df['Product'].value_counts().head(10).index
    df=df[df['Product'].isin(top)]
    return df

def sample(df):
    sampled=df.groupby('Product').sample(n=5000, replace=False, random_state=42)
    return sampled


def main():    

    df =pd.read_csv('data/raw/Consumer_Complaints.csv')
    df.columns

    df=Issue_col(df)
    df.info()

    df=Product_col(df)
    df['Product'].value_counts()

    sampled=sample(df)
    sampled.info()
    sampled.to_csv('data/processed/complaints_sample.csv')
    print()
    print("cleaned dataset file successfully exported into 'data/processed' folder...")


if __name__ == '__main__':
    main()
