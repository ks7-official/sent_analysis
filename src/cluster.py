import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_clustering(data_path='data/merged_data.csv'):
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return

    df = pd.read_csv(data_path).dropna()
    features = df[['Leverage', 'sentiment_score', 'Closed PnL']]
    scaled = StandardScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled)
    df['cluster'] = kmeans.labels_

    os.makedirs("output", exist_ok=True)
    sns.pairplot(df, vars=['Leverage', 'sentiment_score', 'Closed PnL'], hue='cluster', palette='tab10')
    plt.suptitle("🧠 Clustering of Trades", fontsize=16)
    plt.tight_layout()
    plt.savefig("output/clustering_plot.png")
    print("📊 Saved clustering plot to output/clustering_plot.png")

    os.makedirs("data", exist_ok=True)
    df.to_csv('data/clustered_data.csv', index=False)
    print("✅ Clustered data saved to data/clustered_data.csv")

if __name__ == "__main__":
    run_clustering()
