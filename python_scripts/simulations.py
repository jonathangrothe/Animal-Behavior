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
ntargets = 2
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 50
    initialy[a] = 50
initialxt = [25,75]
initialyt = [75,75]

T = 5000
periodicflag = 0
rEgo = 0
rEgoTarget = 0
Egonumber = 1
hColl = -10
rColl = 0
distf = 0
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
    


allocentricFlag = 0
h0s = [0.2]*ntargets
h_b = 0.2
sigma = 0.2
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
    for s in range(1):
        xpoints = []
        ypoints = []
        angles = []
        dists = []
        rng = np.random.default_rng(seed=55)
        ntarg = 10
        while len(xpoints) < ntarg:
            angle = rng.uniform(0,2*np.pi)
            dist = rng.uniform(1,25)
            for p in range(len(angles)):
                a = angles[p]
                d = dists[p]
                angle_unwrapped = np.unwrap([angle,a])
                angle_dist = np.abs(angle_unwrapped[0]-angle_unwrapped[1])
                true_dist = np.sqrt(dist**2+d**2-2*d*dist*np.cos(angle_dist))
                if true_dist < 5:
                    break
                
            angles.append(angle)
            dists.append(dist)
            xpoints.append(50+dist*np.cos(angle))
            ypoints.append(50+dist*np.sin(angle))
        furthest = 3
        print(f"furthest dist: {dists[furthest]}, x:{xpoints[furthest]}, y:{ypoints[furthest]}")
        h0_list = [0.25]*ntarg
        base['h0'] = h0_list
        base['initialxt'] = xpoints
        base['initialyt'] = ypoints
        base['ntargets'] = ntarg
        n_trajpoints = 101
        traj_factor = L/(n_trajpoints-1)
        grid_angles = helpers.trajectory_grid(xpoints,ypoints,n_trajpoints,1,0)
        subfigs, ax1 = plt.subplots(nrows=3,ncols=2,layout='constrained',figsize=(12,12),num=s*2+1)
        ax1 = ax1.flatten()
        fig2, ax2 = plt.subplots(figsize = (8,8),num=s*2+3)

        #sigma_list = [0.05,0.1,0.2,0.4]
        #h0_lists = [[0.15]*10,[0.2]*10,[0.25]*10,[0.3]*10]
        distf_list = [0,1,1]
        adistf_list = [0,1,10]
        #rEgoTarget_list = [0,0,3.5,3.5]
        change_random = {'distf':distf_list,
                         'adistf':adistf_list}
        target_list, time_list, activity_list, x_list, y_list, headings_list, init_heading  = repeated_sims.sample_sims(base,change_random,sample_size,True,True)
        base['h0'] = h0_list
        for g in range(n_trajpoints**2):
            base_x = g % n_trajpoints * traj_factor
            base_y = g // n_trajpoints * traj_factor
            gr_x2 = base_x + 0.5*np.cos(grid_angles[g]) * traj_factor
            gr_y2 = base_y + 0.5*np.sin(grid_angles[g]) * traj_factor
            ax2.arrow(base_x,base_y,gr_x2-base_x,gr_y2-base_y,width=0.005,head_width=0.4,head_length=0.2,color='green')
        
        # plotting expected (approximate) trajectory on top of the grid
        xs = [initialx[0]]
        ys = [initialy[0]]
        start_x = initialx[0]
        start_y = initialy[0]
        for p in range(10000):
            closest_x = round(start_x/traj_factor)
            closest_y = round(start_y/traj_factor)
            ind = (closest_y+1) * n_trajpoints + closest_x+1
            next_x = 0.005*np.cos(grid_angles[ind]) + start_x
            next_y = 0.005*np.sin(grid_angles[ind]) + start_y 
            xs.append(next_x)
            ys.append(next_y)
            start_x = next_x
            start_y = next_y
        #print(f"diffs for combo: {np.array([np.diff(xs_combi),np.diff(ys_combi)])}")
        ax2.plot(xs,ys,c='red')

        for sig in range(len(distf_list)):
            x_settings = x_list[sig*sample_size:(sig+1)*sample_size]
            y_settings = y_list[sig*sample_size:(sig+1)*sample_size]
            sim_met.plot_traj(x_settings,y_settings,xpoints,ypoints,sample_size,ax1[sig*2])
            #ax1[sig*2].plot(xs,ys,c='red',linestyle='--')
            if sig == 0:
                ax1[sig*2].set_title("No distance dependence")
            if sig == 1:
                ax1[sig*2].set_title("Normal distance dependence")
            if sig == 2:
                ax1[sig*2].set_title("Extreme distance dependence")
            if sig == 3:
                ax1[sig*2].set_title("Distance dependent, ego switch within 3.5 units")
            ind = sig*sample_size
            sim_activity = activity_list[ind]
            ax1[sig*2+1].imshow(sim_activity,aspect='auto')

    #subfigs.suptitle("equal targs")
    plt.show()

def run_sims(base,change,change_var,sample_size,traj,activity,ncol,nrow,fig_num):
    target_list, time_list, activity_list, x_list, y_list, headings_list, init_heading  = repeated_sims.sample_sims(base,change,sample_size,traj,activity)
    phases = []
    angles = []
    bif_indices = []
    h0_list = change['h0']
    '''
    theor_angles = []
    for item in h0_list:
        sum_hs = sum(item)
        print(f"bias from item 1: {item[0]/sum_hs*(np.pi/6)} item 2: {item[2]/sum_hs*(-np.pi/6)}")
        angle = item[0]/sum_hs*(np.pi/6)+item[2]/sum_hs*(-np.pi/6) + np.pi/2
        theor_angles.append(angle)
    print(f"theoretical angles: {theor_angles}")
    '''
    fig_init = plt.figure(layout='constrained',figsize=(10,5),num=fig_num+1)
    subfigs_init = fig_init.subfigures(1,1, wspace=0.1,squeeze=False)
    axs0_init = subfigs_init[0][0].subplots(1,2)
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
        heading = init_heading[i].item()
        print(heading)
        x_plot = math.cos(heading)
        y_plot = math.sin(heading)
        color = 'red'
        if target_list[i] == 1:
            color = 'green'
        axs0_init[0].plot([0,x_plot],[0,y_plot],c=color)
        

    n_plots = len(change[change_var])
    ncols = ncol
    nrows = nrow
    fig = plt.figure(layout='constrained',figsize=(ncols*10,nrows*5),num=fig_num)
    subfigs = fig.subfigures(2,1, wspace=0.1)
    axs0 = subfigs[0].subplots(nrows,ncols)
    axs0 = axs0.flatten()
    axs1 = subfigs[1].subplots(nrows,ncols)
    axs1 = axs1.flatten()
    grey_to_blue = ["#D3D3D3", "#A9A9A9", "#708090", "#4682B4", "#000080"]
    cmap = mcolors.LinearSegmentedColormap.from_list("GreyBlue", grey_to_blue)

    for s in range(n_plots):
        sim_met.plot_traj(x_list[s*sample_size:(s+1)*sample_size],y_list[s*sample_size:(s+1)*sample_size],initialxt,initialyt,sample_size,axs0[s],False,[],0,0)
        #theor_angle = theor_angles[s]
        #x2 = 50 + 30 * np.cos(theor_angle)
        #y2 = 50 + 30 * np.sin(theor_angle)
        #axs0[s].plot([50, x2], [50, y2], color='green', linewidth=2, linestyle = '--')
        axs0[s].set_title(f"sigma: {round(sigma_list[s],4)}")
        #axs0[s].set_aspect('equal', adjustable='box')
        target_list_sample = target_list[s*sample_size:(s+1)*sample_size]
        print(f"sim: {s}, targets reached: {target_list_sample}")
        ind = s*sample_size
        sim_activity = activity_list[ind]
        col_min = np.min(sim_activity[:,-5])
        col_max = np.max(sim_activity[:,-5])
        axs1[s].imshow(sim_activity,cmap=cmap,aspect='auto',vmin=col_min,vmax=col_max)
        

# -------- Running the simulation --------
np.random.seed(24)
u0_rand = 0.2*np.random.randn(N,1) 
#u0 = np.zeros((N,1))
#u0[5:31] = 0.2
#u0[0:26] = 0.2
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
        'beta':beta,
        'u0':u0_rand}

plot_neurons = True
plot_trajs = True
sample_size = 1
#h0_first = np.linspace(0.275,0.371,num=6)
h0_list = [[0.27,0.27]]
sigma_list = [0.4]
#u0_list = [10*u0,2*u0,u0,u0*0.5,u0*0.1]
#beta_list = [300,100,60,25,10]
#v0_list = [0.1,0.2,0.3,0.4,0.5,0.6]
#beta_list = [20]*6
u0_list = [u0_rand]
change = {'sigma':sigma_list, 'h0':h0_list, 'u0':u0_list}

# current goal: so it seems like given geometry and the two constant h0s, for most h0s we can determine a sigma that will produce trajectories that minimize the in between zone 
# (ie: we will find the lowest possible sigma that results in the agent reaching the target)
# from there we know there will be the transition from center target to better target. The current goal is estimating when this will occur
# my current theory is that close to the decision it uses a straighter trajectory (aggregation phase), and we can estimate that phase with a line based only on parameters


#run_sims(base,change,'h0',sample_size,plot_trajs,plot_neurons,4,1,1)

run_sims(base,change,'sigma',sample_size,plot_trajs,plot_neurons,2,1,1)
#base['distf'] = 0
#sim_random_points(base,1)


plt.show()