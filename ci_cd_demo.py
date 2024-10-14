import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

save_plot = Path(__file__).parent/'plots'

fig = plt.figure(figsize=(10, 6))
x_pts = ['Random Forest', 'XG Boost']
y_pts = [0.698, 0.83]
colors = ['red', 'blue']

plt.bar(x=x_pts, height=y_pts, label=x_pts, color=colors)
plt.legend()
plt.ylim((0,1))
plt.show()

plt.savefig(save_plot/'bar_plot_2.png')