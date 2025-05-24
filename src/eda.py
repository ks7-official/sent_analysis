import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def plot_trades_by_sentiment(df):
    sns.countplot(data=df, x='classification', palette='coolwarm')
    plt.title('📊 Trader Actions Across Market Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Number of Trades')
    plt.tight_layout()
    plt.savefig('output/sentiment_trade_count.png')
    plt.clf()

def plot_average_pnl_by_sentiment(df):
    avg_pnl = df.groupby('classification')['Closed PnL'].mean().reset_index()
    sns.barplot(data=avg_pnl, x='classification', y='Closed PnL', palette='viridis')
    plt.title('💰 Average PnL by Market Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Average Closed PnL')
    plt.tight_layout()
    plt.savefig('output/average_pnl_by_sentiment.png')
    plt.clf()

def plot_leverage_distribution(df):
    sns.boxplot(data=df, x='classification', y='Leverage', palette='pastel')
    plt.title('📈 Leverage Distribution by Market Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Leverage')
    plt.tight_layout()
    plt.savefig('output/leverage_by_sentiment.png')
    plt.clf()

def plot_side_vs_sentiment(df):
    sns.countplot(data=df, x='classification', hue='Side', palette='Set2')
    plt.title('🟢 BUY vs 🔴 SELL under Different Market Sentiments')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Trade Count')
    plt.legend(title='Trade Side')
    plt.tight_layout()
    plt.savefig('output/trade_side_by_sentiment.png')
    plt.clf()

def plot_pnl_distribution_violin(df):
    sns.violinplot(data=df, x='classification', y='Closed PnL', palette='muted', inner='quartile')
    plt.title('🎻 Closed PnL Distribution by Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Closed PnL')
    plt.tight_layout()
    plt.savefig('output/pnl_violin_by_sentiment.png')
    plt.clf()

def plot_sentiment_over_time(df):
    df_sorted = df.sort_values('datetime')
    sns.lineplot(data=df_sorted, x='datetime', y='sentiment_score', color='orange')
    plt.title('🕒 Sentiment Score Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('output/sentiment_score_time.png')
    plt.clf()

def plot_correlation_heatmap(df):
    numeric_df = df[['Closed PnL', 'Leverage', 'sentiment_score']].dropna()
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('🔥 Correlation Between Numerical Features')
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png')
    plt.clf()

if __name__ == "__main__":
    df = load_data('data/merged_data.csv')
    plot_trades_by_sentiment(df)
    plot_average_pnl_by_sentiment(df)
    plot_leverage_distribution(df)
    plot_side_vs_sentiment(df)
    plot_pnl_distribution_violin(df)
    plot_sentiment_over_time(df)
    plot_correlation_heatmap(df)
    print("✅ All EDA plots saved to /output folder")
