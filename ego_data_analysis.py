import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simulate_ringattractor as sim_ra

positions = pd.read_csv("positions_ego_df.csv")
activity = pd.read_csv("activity_ego_df.csv")
headings = pd.read_csv("headings_ego_df.csv")

headings_diff = np.zeros(len(headings.iloc[0,:]))
headings_diff[1:] = np.diff(headings.iloc[0,:])
headings_oscilate = np.abs(headings_diff) > np.pi

indices = [i for i, val in enumerate(headings_oscilate) if val]
indices.reverse()

plt.figure(1)
plt.plot(positions['x diff'])

plt.figure(2)
plt.plot(positions['y diff'])

plt.figure(3)
plt.plot(headings.iloc[0,:])

# we need to find the final time the heading oscilates

max_x_diff = np.max(positions['x diff'])
time_max_x_diff = np.argmax(positions['x diff'])

last_oscilation = 0
for item in indices:
    if item < time_max_x_diff: 
        last_oscilation = item
        break


range_stop = time_max_x_diff +100

activity_range = activity.iloc[:,time_max_x_diff:range_stop]


for i in range(325,330):
    curr_heading = headings.iloc[0,i]
    neuron_heading = (100/(2*np.pi))*curr_heading
    if curr_heading > np.pi:
        curr_heading = curr_heading - 2*np.pi
        neuron_heading = (100/(2*np.pi))*curr_heading+100
    curr_x = positions.iloc[i,0]
    curr_y = positions.iloc[i,1]
    activity_at_heading_direction = activity.iloc[0,i]
    max_activity = np.max(activity.iloc[:,i])
    argmax_activity = np.argmax(activity.iloc[:,i])
    angle_to_top = np.atan((80-curr_y)/(80-curr_x))-curr_heading
    angle_to_bottom = np.atan((20-curr_y)/(80-curr_x))-curr_heading
    top_bump = (100/(2*np.pi))*angle_to_top
    bottom_bump = (100/(2*np.pi))*angle_to_bottom+100

    cx_d = 0
    cy_d = 0
    for n in range(100):
        val = sim_ra.fAct(activity.iloc[n,i],100) 
        if val < 0:
            val = 0
        alpharing = np.linspace(curr_heading,curr_heading+2*np.pi,101)
        alpharing = alpharing[:-1]
        alpharing = np.mod(alpharing, 2*np.pi)
        cx_d = cx_d + val * np.cos(alpharing[n])
        cy_d = cy_d + val * np.sin(alpharing[n])
    print(f"time: {i}")
    print(f"center x: {cx_d}, center y: {cy_d}")
    print(f"heading: {curr_heading}, neuron as heading: {neuron_heading}, x position: {curr_x}, y position: {curr_y}")
    print(f"angle to top: {angle_to_top}, estimated top bump: {top_bump}, angle to bottom: {angle_to_bottom}, estimated bottom bump: {bottom_bump}")
    print(f"activity at heading: {activity_at_heading_direction}, max activity: {max_activity}, neuron with max activity: {argmax_activity}")
    



'''
alpharing = np.linspace(0,2*np.pi,101)
alpharing = alpharing[:-1]
print(alpharing)
for index in range(len(headings.iloc[0,:])-1):
    cx_d = 0
    cy_d = 0
    for i in range(100):
        val = sim_ra.fAct(activity.iloc[i,index+1],100) 
        if val < 0:
            val = 0
        cx_d = cx_d + val * np.cos(alpharing[i])
        cy_d = cy_d + val * np.sin(alpharing[i])
    newAngle = np.atan2(cx_d,cy_d)
    print(f"cx: {cx_d}, cy: {cy_d}, angle: {newAngle}")
    alpharing = np.mod(alpharing - headings.iloc[0,index] + newAngle,2*np.pi)

    # if index is in the range of decision, what do we want? 
    # new angle
    # alpharing (just the first entry is ok)
'''


plt.show()

