import matplotlib.pyplot as plt
import numpy as np
scores = np.load("scores.npy")
diff = np.load("diff.npy")
print(scores)
print(diff)
exit()
plt.plot(scores)
plt.plot(diff)
plt.show()
