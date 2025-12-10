import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
train_log = './data/training_log.csv'
pos_log = './data/positions_log.csv'

def show_dashboard():
    if not os.path.exists(train_log) or not os.path.exists(pos_log):
        print("Data files not found in ./data/")
        return

    # Load Data
    df_train = pd.read_csv(train_log)
    df_pos = pd.read_csv(pos_log)

    # Setup Plot
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(16, 12), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    # 1. Score History
    ax1 = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=df_train, x='Game_No', y='Score', ax=ax1, label='Score', alpha=0.6)
    sns.lineplot(data=df_train, x='Game_No', y='Record', ax=ax1, label='Record', linewidth=2)
    ax1.set_title("Score History")
    ax1.legend()

    # 2. Death Analysis
    ax2 = fig.add_subplot(gs[0, 1])
    death_counts = df_train['Death_Reason'].value_counts()
    if len(death_counts) > 0:
        ax2.pie(death_counts, labels=death_counts.index, autopct='%1.1f%%', startangle=140)
    ax2.set_title("Death Reasons")

    # 3. Efficiency (Avg Steps)
    ax3 = fig.add_subplot(gs[1, 0])
    # Convert string metrics to float if needed, though they should be float from CSV
    sns.scatterplot(data=df_train, x='Score', y='Avg_Steps', ax=ax3, hue='Death_Reason', alpha=0.7)
    ax3.set_title("Efficiency: Steps per Food vs Score")

    # 4. Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    # 2D Histogram
    h = ax4.hist2d(df_pos['X'], df_pos['Y'], bins=(32, 24), cmap='hot', cmin=1)
    fig.colorbar(h[3], ax=ax4, label='Visits')
    ax4.set_title("Snake Heatmap")
    ax4.invert_yaxis() # Match pygame coordinates

    plt.suptitle("Snake AI Analytics Dashboard", fontsize=16)
    plt.show()

if __name__ == '__main__':
    show_dashboard()
