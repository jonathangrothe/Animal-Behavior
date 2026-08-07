import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from . import simulation_metrics as sim_met
from . import repeated_sims
from . import simulate_ringattractor as sim_ra
from helper_functions import helpers


L = 100
ntargets = 3
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 50
    initialy[a] = 50
initialxt = [35,50,65]
initialyt = [50+15*np.sqrt(3),80,50+15*np.sqrt(3)]

T = 5000
periodicflag = 0
rEgo = 0
rEgoTarget = 0
Egonumber = 1
hColl = -10
rColl = 0
distf = 1
adistf = 1
dt = 0.1
v0 = 0.05
v0t = np.zeros(ntargets)
for i in range(ntargets):
    v0t[i] = 0

N = 100
nu = 0.5
theta = np.linspace(0,2*np.pi,N+1)
theta = theta[:-1]
J = np.zeros((N,N))
for i in range(N):
    deltah = np.abs(theta - theta[i])
    deltah = np.pi-np.abs(np.pi-deltah)
    J[i,:] = np.cos(np.pi*(deltah/np.pi)**nu)
    J[i,i] = 0.0
    J = np.squeeze(J)

allocentricFlag = 1
h0s = [0.21903,0.21903,0.21904]
h_b = 0.2
sigma = 0.05
beta = 100 

# --- trying out the non stopping situation --- - turn this and the traditional simulations into functions so its easier to keep it organized
def sim_random_points(base,sample_size): 
    # From playing around with this I know that it is often possible for the agent to travel between targets, 
    # it would be nice if it would be possible to get a good guess for sigma/h0 that will allow it to move between targets given the geometry
    # I hypothesize that this will be the most sensitive 'best' version of the model at detecting small differences
    # (I will have to test this), because it clearly has the capacity to sustain/shift between bumps for individual points
    # Q: will this be easier if we increase number of neurons ? (so that we don't aggregate two bumps within 2pi/100 degrees of each other) 

    # it would be a lot easier to gaurantee that they're all approximately different angles from the start point
    # eventually we will need to be able to calculate how points are aggregated
    for s in range(2):
        xpoints = []
        ypoints = []
        angles = []
        dists = []
        rng = np.random.default_rng()
        dir_x = 0 
        dir_y = 0
        while len(xpoints) < 10:
            angle = rng.uniform(0,2*np.pi)
            dist = rng.uniform(5,45)
            for a in angles:
                angle_unwrapped = np.unwrap([angle,a])
                angle_dist = np.abs(angle_unwrapped[0]-angle_unwrapped[1])
                if angle_dist < np.pi/10:
                    break
            dists.append(dist)
            angles.append(angle)
            xpoints.append(50+dist*np.cos(angle))
            ypoints.append(50+dist*np.sin(angle))
        base['h0'] = [0.25]*10
        print(f"base: {base['h0']}")
        base['initialxt'] = xpoints
        base['initialyt'] = ypoints
        base['ntargets'] = 10
        dists = np.array(dists)
        weights = np.exp(-1 * dists / 100)
        x_mean = np.sum(weights * np.cos(angles))
        y_mean = np.sum(weights * np.sin(angles))
        weighted_mean_angle = np.atan2(y_mean, x_mean) # once we've calculated this, move one unit in this direction, recalculate everything
        print(f"initial weighted mean: {weighted_mean_angle}")
        approx_traj = [weighted_mean_angle]
        prev_angle = weighted_mean_angle
        prev_x = 50
        prev_y = 50
        x_traj = [50]
        y_traj = [50]
        for a in range(30):
            new_x = prev_x + 5*np.cos(prev_angle)
            new_y = prev_y + 5*np.sin(prev_angle)
            x_traj.append(new_x)
            y_traj.append(new_y)
            # recalculate all the distances and angles and stuff
            new_x_sum = 0
            new_y_sum = 0
            for p in range(10):
                # calculate distance and weight
                x_diff = xpoints[p]- new_x
                y_diff = ypoints[p] - new_y
                new_dist = np.sqrt((x_diff)**2+(y_diff)**2)
                new_weight = np.exp(-1*new_dist/100)
                new_angle = np.atan2(y_diff,x_diff)
                #print(f"new angle to point {p}: {new_angle}")
                new_x_sum += new_weight * np.cos(new_angle)
                new_y_sum += new_weight * np.sin(new_angle)
            new_overall_angle = np.atan2(new_y_sum,new_x_sum)
            approx_traj.append(new_overall_angle)
            #print(f"new overall angle: {new_overall_angle}")
            prev_angle = new_overall_angle
            prev_x = new_x
            prev_y = new_y 
        final_x = prev_x + 5*np.cos(prev_angle)
        final_y = prev_y + 5*np.sin(prev_angle)
        x_traj.append(final_x)
        y_traj.append(final_y)
        grid_angles = helpers.trajectory_grid(xpoints,ypoints)
        #print(f"angles of approximated trahectory: {approx_traj}")
        fig = plt.figure(layout='constrained',figsize=(7,14),num=s+1)
        subfigs = fig.subfigures(4,1, wspace=0.1)
        axs0 = subfigs[0].subplots(1,2)
        axs0 = axs0.flatten()
        axs1 = subfigs[1].subplots(1,2)
        axs1 = axs1.flatten()
        axs2 = subfigs[2].subplots(1,2)
        axs2 = axs2.flatten()
        axs3 = subfigs[3].subplots(1,2)
        axs3 = axs3.flatten()
        
        sigma_sim = []
        for sig in range(4):
            sigma_sim.append(0.03 + sig*0.15)

        change_random = {'sigma':sigma_sim}
        print(f"base: h0: {base['h0']}")
        target_list, time_list, activity_list, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change_random,sample_size,True, True)
        for sig in range(len(sigma_sim)):
            axs = 0
            if sig == 0:
                axs = axs0
            if sig == 1:
                axs = axs1
            if sig == 2:
                axs = axs2
            if sig == 3:
                axs = axs3
            #for pt in range(len(x_traj)-1):
                #axs[0].plot([x_traj[pt], x_traj[pt+1]], [y_traj[pt], y_traj[pt+1]], color='green', linewidth=1, linestyle = '--')
            x_settings = x_list[sig*sample_size:(sig+1)*sample_size]
            y_settings = y_list[sig*sample_size:(sig+1)*sample_size]
            sim_met.plot_traj(x_settings,y_settings,xpoints,ypoints,sample_size,axs[0])
            ind = sig*sample_size
            sim_activity = activity_list[ind]
            axs[1].imshow(sim_activity,aspect='auto')
            for g in range(441):
                base_x = g % 21 * 5
                base_y = g // 21 * 5
                gr_x2 = base_x + 2*np.cos(grid_angles[g])
                gr_y2 = base_y + 2*np.sin(grid_angles[g])
                #axs[2].plot([base_x, gr_x2], [base_y, gr_y2], color='red', linewidth=1, linestyle = '--')
                axs[0].arrow(base_x,base_y,gr_x2-base_x,gr_y2-base_y,width=0.05,head_width=0.6,head_length=0.6,color='green')

        
    
        
    plt.show()

def run_sims(base,change,change_var,sample_size,traj,activity,ncol,nrow):
    target_list, time_list, activity_list, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,traj,activity)
    phases = []
    angles = []
    bif_indices = []
    h0_list = change['h0']
    theor_angles = []
    for item in h0_list:
        sum_hs = sum(item)
        print(f"bias from item 1: {item[0]/sum_hs*(np.pi/6)} item 2: {item[2]/sum_hs*(-np.pi/6)}")
        angle = item[0]/sum_hs*(np.pi/6)+item[2]/sum_hs*(-np.pi/6) + np.pi/2
        theor_angles.append(angle)
    print(f"theoretical angles: {theor_angles}")
    for i in range(len(target_list)):
        curr_activity = activity_list[i]
        curr_headings = headings_list[i][0]
        curr_xpos = x_list[i:(i+1)][0]
        curr_ypos = y_list[i:(i+1)][0]
        target = target_list[i]
        indices = [0]
        bifurcation_angles = [np.pi/2]
        if target != -1:
            indices, bifurcation_angles = sim_met.get_bifurcation_angle(curr_xpos,curr_ypos,curr_headings)
        phase = sim_met.get_bump_type(curr_activity)
        angles.append(bifurcation_angles)
        bif_indices.append(indices)
        phases.append(phase)

    n_plots = len(change[change_var])
    ncols = ncol
    nrows = nrow
    fig = plt.figure(layout='constrained',figsize=(ncols*3.5,nrows*6))
    subfigs = fig.subfigures(2,1, wspace=0.1)
    axs0 = subfigs[0].subplots(nrows,ncols)
    axs0 = axs0.flatten()
    axs1 = subfigs[1].subplots(nrows,ncols)
    axs1 = axs1.flatten()
    grey_to_blue = ["#D3D3D3", "#A9A9A9", "#708090", "#4682B4", "#000080"]
    cmap = mcolors.LinearSegmentedColormap.from_list("GreyBlue", grey_to_blue)

    for s in range(n_plots):
        sim_met.plot_traj(x_list[s*sample_size:(s+1)*sample_size],y_list[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,axs0[s],False,[[100,1000,2000]],0)
        theor_angle = theor_angles[s]
        x2 = 50 + 30 * np.cos(theor_angle)
        y2 = 50 + 30 * np.sin(theor_angle)
        #axs0[s].plot([50, x2], [50, y2], color='green', linewidth=2, linestyle = '--')
        axs0[s].set_title(f"sigma: {round(sigma_list[s],4)}, h0: {h0_list[s]}, beta: {beta}")
        axs0[s].set_aspect('equal', adjustable='box')
        target_list_sample = target_list[s*sample_size:(s+1)*sample_size]
        print(f"sim: {s}, targets reached: {target_list_sample}")
        ind = s*sample_size
        sim_activity = activity_list[ind]
        col_min = np.min(sim_activity[:,40:])
        col_max = np.max(sim_activity[:,40:])
        axs1[s].imshow(sim_activity,cmap=cmap,aspect='auto',vmin=col_min,vmax=col_max)

# -------- Running the simulation --------

base = {'N':N,
        'L':L,
        'ntargets':ntargets,
        'nagents':nagents,
        'initialx':initialx,
        'initialy':initialy,
        'initialxt':initialxt,
        'initialyt':initialyt,
        'T':T,
        'periodicFlag':periodicflag,
        'rEgo':rEgo,
        'rEgoTarget':rEgoTarget,
        'Egonumber':Egonumber,
        'hColl':hColl,
        'rColl':rColl,
        'distf':distf,
        'adistf':adistf,
        'dt':dt,
        'v0':v0,
        'v0t':v0t,
        'nu':nu,
        'J':J,
        'allocentricFlag':allocentricFlag,
        'h0':h0s,
        'h_b':h_b,
        'sigma':sigma,
        'beta':beta}

plot_neurons = True
plot_trajs = True
sample_size = 5
#h0_first = np.linspace(0.275,0.371,num=6)
h0_list = [[0.306,0.275,0.275],[0.309,0.275,0.275],[0.312,0.275,0.275],[0.315,0.275,0.275],[0.318,0.275,0.275],[0.321,0.275,0.275]]
sigma_list = [0.09]*6
change = {'sigma':sigma_list, 'h0':h0_list}

# current goal: so it seems like given geometry and the two constant h0s, for most h0s we can determine a sigma that will produce trajectories that minimize the in between zone 
# (ie: we will find the lowest possible sigma that results in the agent reaching the target)
# from there we know there will be the transition from center target to better target. The current goal is estimating when this will occur
# my current theory is that close to the decision it uses a straighter trajectory (aggregation phase), and we can estimate that phase with a line based only on parameters


#run_sims(base,change,'h0',sample_size,plot_trajs,plot_neurons,3,2)
sim_random_points(base,1)
plt.show()