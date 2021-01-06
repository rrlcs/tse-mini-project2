import numpy as np
import re
# import matplotlib
import matplotlib.pyplot as plt

with open("lossess", "r") as f:
    data = f.read()
data = re.findall("\d+\.\d+", data)
data = np.array(list(map(float, data)))
print(data.size)
iterations = np.array([500*i for i in range(1, data.size+1)])
plt.plot(iterations, data)
plt.savefig("lossplot.png")
