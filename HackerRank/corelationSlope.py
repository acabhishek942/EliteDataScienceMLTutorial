import numpy as np

x = np.array([[1,15],[1,12],[1,8],[1,8],[1,7],[1,7],[1,7],[1,6],[1,5],[1,3]])
y = np.array([[10],[25],[17],[11],[13],[17],[20],[13],[9],[15]])
inverse = np.linalg.inv((x.T).dot(x))
slope=(inverse.dot(x.T)).dot(y)
print(round(float(slope[1,0]),3))
