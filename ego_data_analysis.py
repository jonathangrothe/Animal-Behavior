import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simulate_ringattractor as sim_ra
from scipy.signal import find_peaks
from scipy.integrate import simpson

'''
positions = pd.read_csv("positions_ego_df.csv")
activity_df = pd.read_csv("activity_ego_df.csv")
activity_to_insert = activity_df.iloc[0:4,1:]
activity = pd.concat([activity_df.iloc[:,1:],activity_to_insert]).reset_index(drop=True)
headings = pd.read_csv("headings_ego_df.csv")


headings_diff = np.zeros(len(headings.iloc[0,:]))
headings_diff[1:] = np.diff(headings.iloc[0,:])
headings_oscilate = np.abs(headings_diff) > np.pi
for index in range(len(headings.iloc[0,:])):
    item = headings.iloc[0,index]
    if item > np.pi:
        new_heading = item - 2*np.pi
        headings.iloc[0,index] = new_heading
        

indices = [i for i, val in enumerate(headings_oscilate) if val]
indices.reverse()

max_x_diff = np.max(positions['x diff'])
time_max_x_diff = np.argmax(positions['x diff'])

last_oscilation = 0
for item in indices:
    if item < time_max_x_diff: 
        last_oscilation = item
        break


range_stop = time_max_x_diff +100

activity_range = activity.iloc[:,time_max_x_diff:range_stop]
'''

'''
plt.figure(1)
plt.plot(activity.iloc[:,354])
plt.title("activity at time 354")

plt.figure(2)
plt.plot(activity.iloc[:,355])
plt.title("activity at time 355")

plt.figure(3)
plt.plot(activity.iloc[:,356])
plt.title("activity at time 356")
'''

def plot_area(activity_df,fig,ylim):
    '''
    a function that plots the area under each bump at each time points
    parameters:
    activity: a dataframe of activity
    '''
    activity_to_insert = activity_df.iloc[0:4,1:]
    activity = pd.concat([activity_df.iloc[:,1:],activity_to_insert]).reset_index(drop=True)
    bump_differences = []
    bump1_areas = []
    bump2_areas = []
    start_ind = 0
    finish_ind = len(activity.iloc[0,:])
    for i in range(start_ind,finish_ind):
        indices_bumps, _ = find_peaks(activity.iloc[:,i])
        true_bumps = [x for x in indices_bumps if x <= 100]
        negative_bumps, _ = find_peaks(-activity.iloc[:,i])
        true_troughs = [x for x in negative_bumps if x <= 100]
        for index in range(len(true_bumps)): 
            if true_bumps[index] == 100:
                true_bumps[index] = 0
                true_bumps.reverse()
        for index in range(len(true_troughs)): 
            if true_troughs[index] == 100:
                true_troughs[index] = 0

        if len(true_bumps) == 2:
            if true_bumps[1] >= 97 and true_bumps[0] >= 20:
                true_bumps.reverse()
        #print(f"time: {i}, bump indices: {true_bumps}, troughs indices: {true_troughs}")
        base_activity = np.max(activity.iloc[negative_bumps,i])
        #print(f"bump_base_activity: {base_activity}")
        bump_areas = []
        if len(true_bumps) != 2:
            bump1_areas.append(np.nan)
            bump2_areas.append(np.nan)
            bump_differences.append(np.nan)
            #print(f"NOT A NUMBER DETECTED at {i}")
        if len(true_bumps) == 2:
            for item in true_bumps: 
                bump_activity = activity.iloc[item,i]
                activity_left = bump_activity
                activity_right = bump_activity
                left_index = item-1
                right_index = item+1
                while activity_left > base_activity:
                    left_index = left_index -1
                    if left_index <= -1:
                        left_index = 99
                    activity_left = activity.iloc[left_index,i]
                    #print(f"entered, at {left_index}, activity left: {activity_left}")
                while activity_right > base_activity:
                    right_index = right_index +1
                    if right_index >= 100:
                        right_index = 0
                    activity_right = activity.iloc[right_index,i]
                    #print(f"entered, at {right_index}, activity right: {activity_right}")
                #print(f"bump: {item}, left index of bump: {left_index}, right index of bump: {right_index}")
            # now we have indices we need to integrate with base_activity as 0
            # and we have to roll over anything that is
                #print(f"left index: {left_index}")
                #print(f"right index: {right_index}")
                if right_index >= left_index:
                    if right_index not in true_troughs: 
                        right_index = right_index +1
                    if left_index not in true_troughs:
                        left_index = left_index -1
                    if right_index == 100:
                        right_index = 0
                    if left_index == -1:
                        left_index = 99
                    activity_for_integral = activity.iloc[left_index:right_index,i]
                if right_index < left_index:
                    #print(f"left activity: {activity.iloc[left_index:100,i]}")
                    #print(f"right activity: {activity.iloc[0:right_index+1,i]}")
                    width = 100 - left_index
                    right_index = width + right_index
                    rolled = np.roll(activity.iloc[0:100,i], width,axis=0)
                    activity_for_integral = rolled[0:right_index+1]
                activity_for_integral = activity_for_integral - base_activity
                #print(f"activity for integral: {activity_for_integral}")
                if len(activity_for_integral) > 1 and i > 100:
                    area_simpson = simpson(activity_for_integral)
                else:
                    area_simpson = 0
                #print(f"bump: {item}, total bump area: {area_simpson}")
                bump_areas.append(area_simpson)
            bump1_areas.append(bump_areas[0])
            bump2_areas.append(bump_areas[1])
            #bump_differences.append(bump_areas[0]/bump_areas[1])

    # NEED TO CHANGE THIS SO IT HANDLES THE GAPS WHEN THERE ARE NOT EXACTLY TWO WELL DEFINED BUMPS
    #plt.figure(figsize=(15,7))
    #plt.figure(4)
    #fig.set_title("Area under each bump, example h0: 0.26,0.27, beta: 20, egocentic case")
    fig.set_ylim(ylim)
    fig.plot(bump1_areas,color='blue')
    fig.plot(bump2_areas,color='red')
    fig.set_xlabel("time")
    fig.set_ylabel("area under each bump")


plt.show()

