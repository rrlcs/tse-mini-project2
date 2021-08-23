import numpy as np
import re
import matplotlib.pyplot as plt

# specify loss file name
with open("lossess", "r") as f:
    data = f.read()
data = re.findall("\d+\.\d+", data)
data = np.array(list(map(float, data)))
iterations = np.array([i for i in range(1, data.size)])
plt.plot(iterations, data[1:])
# specify file name for plot image
plt.savefig("train_lossplot.png")
