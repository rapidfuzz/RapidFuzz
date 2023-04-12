from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv("temp/result.csv")

results *= 1000 * 1000
results["x_axis"] /= 1000 * 1000


ax = results.plot(x="x_axis")

# plt.xticks(list(range(0, 64*20+1, 64)))

plt.title("Performance comparison of the \nDamerauLevenshtein similarity in different libraries")
plt.xlabel("string length [in characters]")
plt.ylabel("runtime [μs]")
ax.set_xlim(xmin=0)
# ax.set_ylim(bottom=0)
# ax.set_yscale('log')
plt.grid()
plt.show()
