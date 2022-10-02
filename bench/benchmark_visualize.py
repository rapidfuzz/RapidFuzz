import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("results/levenshtein_damerau.csv")

df *= 1000 * 1000
df["length"] /= 1000 * 1000


ax=df.plot(x="length")

plt.xticks(list(range(0, 257, 64)))

plt.title("Performance comparision of the \nDamerauLevenshtein similarity in different libraries")
plt.xlabel("string length [in characters]")
plt.ylabel("runtime [μs]")
ax.set_xlim(xmin=0)
ax.set_ylim(bottom=0)
plt.grid()
plt.show()
