import pandas as pd
import numpy as np

def load_and_clean_sentiment(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['label'] = df['classification'].map({'Fear': 0, 'Greed': 1})
    df = df[['datetime', 'value', 'label']]
    df.rename(columns={'value': 'sentiment_score'}, inplace=True)
    return df

def load_and_clean_trades(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['Timestamp IST'], format="%d-%m-%Y %H:%M")
    df['Closed PnL'] = pd.to_numeric(df['Closed PnL'], errors='coerce')
    df['Leverage'] = df['Size USD'] / df['Execution Price']
    df = df[['Account', 'Coin', 'Execution Price', 'Size USD', 'Side', 'datetime', 'Closed PnL', 'Leverage']]
    return df

def merge_datasets(trades_df, sentiment_df):
    sentiment_df = sentiment_df.set_index('datetime').resample('H').ffill().reset_index()
    trades_df['datetime_rounded'] = trades_df['datetime'].dt.floor('H')
    merged = pd.merge(trades_df, sentiment_df, left_on='datetime_rounded', right_on='datetime', how='left')
    merged = merged.drop(columns=['datetime_rounded', 'datetime_y']).rename(columns={'datetime_x': 'datetime'})
    return merged

def classify_sentiment(score):
    if score >= 60:
        return "Greed"
    elif score >= 40:
        return "Neutral"
    else:
        return "Fear"

if __name__ == "__main__":
    sentiment = load_and_clean_sentiment('data/fear_greed_index.csv')
    trades = load_and_clean_trades('data/historical_data.csv')
    merged = merge_datasets(trades, sentiment)
    merged["classification"] = merged["sentiment_score"].apply(classify_sentiment)
    merged.to_csv('data/merged_data.csv', index=False)
    print("✅ Merged data saved to data/merged_data.csv")
