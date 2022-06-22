import numpy as np
import matplotlib.pyplot as plt

slope = 2

O = [10,10]

x = np.linspace(0,20)
y = (x - O[0]) * slope + O[1]

x_p = 15
y_p = 10
y_opp = -1./slope*(x-x_p)+y_p

x_x = (slope ** 2.0 * O[0] - slope * O[1] + slope * y_p + x_p) / (1. + slope ** 2.)
y_x = (x_x - O[0]) * slope + O[1]

import time
start_time = time.time()
goopy = np.zeros(1000000)
for i in range(0,25):
	for j in range(0,1000000):
		goopy[j] += 100.
print (time.time() - start_time)

plt.plot(x,y,color='k')
plt.scatter(O[0],O[1],color='r')
plt.scatter(x_p,y_p,color='g')
plt.plot(x,y_opp,color='k',linestyle='--')
plt.scatter(x_x,y_x,color='b')
plt.xlim(0,20)
plt.ylim(0,20)
plt.gca().set_aspect('equal', adjustable='box')
#plt.show()
