import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simulate_ringattractor as sim_ra
from scipy.signal import find_peaks
from scipy.integrate import simpson

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

'''
plt.figure(1)
plt.plot(positions['x diff'],color='red',label='x diff')
plt.plot(positions['y diff'],color='blue',label='y diff')

plt.figure(2)
plt.plot(headings.iloc[0,:])
'''

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

#print(f"activity at neuron 49: {activity.iloc[49,:]}")
#print(f"activity at time 363: {activity.iloc[49,363]}, activity at time 364: {activity.iloc[49,364]}")
indices, _ = find_peaks(activity.iloc[49,:])
#print(f"local maxima of neuron 49: {indices}")


plt.figure(1)
plt.plot(activity.iloc[:,200])
plt.title("activity at time 200")


bump_differences = []
for i in range(100,400):
    indices_bumps, _ = find_peaks(activity.iloc[:,i])
    true_bumps = [x for x in indices_bumps if x <= 100]
    negative_bumps, _ = find_peaks(-activity.iloc[:,i])
    true_troughs = [x for x in negative_bumps if x <= 100]
    for index in range(len(true_bumps)): 
        if true_bumps[index] == 100:
            true_bumps[index] = 0
    for index in range(len(true_troughs)): 
        if true_troughs[index] == 100:
            true_troughs[index] = 0
    print(f"time: {i}, bump indices: {true_bumps}, troughs indices: {true_troughs}")
    # find the trough that is closer between the bumps ? 
    base_activity = np.max(activity.iloc[negative_bumps,i])
    #print(f"bump_base_activity: {base_activity}")
    bump_areas = []
    if len(true_bumps) ==2:
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
            area_simpson = simpson(activity_for_integral)
            print(f"bump: {item}, total bump area: {area_simpson}")
            bump_areas.append(area_simpson)
        bump_differences.append(bump_areas[0]/bump_areas[1])

        
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
    '''
    if i in [144,364,366]:
        print(f"time: {i}")
        print(f"center x: {cx_d}, center y: {cy_d}")
        print(f"heading: {curr_heading}, neuron as heading: {neuron_heading}, x position: {curr_x}, y position: {curr_y}")
        print(f"angle to top: {angle_to_top}, estimated top bump: {top_bump}, angle to bottom: {angle_to_bottom}, estimated bottom bump: {bottom_bump}")
        print(f"activity at heading: {activity_at_heading_direction}, max activity: {max_activity}, neuron with max activity: {argmax_activity}")
        curr_activity = activity.iloc[:,i]
        next_activity = activity.iloc[:,i+1]
        dtdu = next_activity - curr_activity
        print(f"current: {np.array(curr_activity)}")
        print(f"next: {np.array(next_activity)}")
        print(f"change: {np.array(dtdu)}")
        '''
print(bump_differences)
plt.figure(2)
plt.plot(bump_differences)
# NEED TO CHANGE THIS SO IT HANDLES THE GAPS WHEN THERE ARE NOT EXACTLY TWO WELL DEFINED BUMPS


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

