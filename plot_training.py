
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read SAC training logs
df = pd.read_csv("logs/monitor_sac.csv", comment="#")

# Calculate rolling average reward
window = 20
df["rolling_reward"] = df["r"].rolling(
    window=window, min_periods=1
).mean()

# Plot episode rewards
plt.figure(figsize=(12, 6))
plt.plot(df["r"], alpha=0.3, label="Episode Reward")
plt.plot(
    df["rolling_reward"],
    linewidth=2,
    label=f"{window}-Episode Moving Average"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SAC Training Performance - UR5 Grasping")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/sac_training_curve.png", dpi=300)
plt.show()
