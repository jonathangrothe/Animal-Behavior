import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

positions = pd.read_csv("positions_ego_df.csv")
activity = pd.read_csv("activity_ego_df.csv")

plt.figure(1)
plt.plot(positions['x diff'])

plt.figure(2)
plt.plot(positions['y diff'])

max_x_diff = np.max(positions['x diff'])
time_max_x_diff = np.argmax(positions['x diff'])
print(max_x_diff)
print(time_max_x_diff)

range_stop = time_max_x_diff +100

activity_range = activity.iloc[:,time_max_x_diff:range_stop]

# find aggregate positions for each one? tough because ring attractor shifts...

'''
cx_d = 0
cy_d = 0
for i in range(N):
    val = fAct(uArray[i,a,tstep+1],beta) 
    if val < 0:
        val = 0
    cx_d = cx_d + val * np.cos(alpharing[a,i])
    cy_d = cy_d + val * np.sin(alpharing[a,i])
'''
plt.show()

