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
initialxt = [50-15*np.sqrt(3),50,50+15*np.sqrt(3)]
initialyt = [35,80,35]

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
def sim_random_points(): 
    # From playing around with this I know that it is often possible for the agent to travel between targets, 
    # it would be nice if it would be possible to get a good guess for sigma/h0 that will allow it to move between targets given the geometry
    # I hypothesize that this will be the most sensitive 'best' version of the model at detecting small differences
    # (I will have to test this), because it clearly has the capacity to sustain/shift between bumps for individual points
    # Q: will this be easier if we increase number of neurons ? (so that we don't aggregate two bumps within 2pi/100 degrees of each other) 

    # it would be a lot easier to gaurantee that they're all approximately different angles from the start point
    # eventually we will need to be able to calculate how points are aggregated
    for s in range(3):
        xpoints = []
        ypoints = []
        angles = []
        rng = np.random.default_rng()
        while len(xpoints) < 10:
            angle = rng.uniform(-np.pi,np.pi)
            dist = rng.uniform(5,45)
            for a in angles:
                angle_dist = np.abs(angle-a)
                if angle_dist < np.pi/10:
                    print("rejected")
                    break

            angles.append(angle)
            print(f"angle: {angle}, x: {50+dist*np.cos(angle)}, y: {50+dist*np.sin(angle)}")
            xpoints.append(50+dist*np.cos(angle))
            ypoints.append(50+dist*np.sin(angle))
            print(f"angle to new point: {angle}")
        h0s_same = [0.25]*10
        print("ONE SET OF POINTS")
        print(f"mean x: {np.mean(xpoints)}, std x: {np.std(xpoints)}")
        print(f"mean y: {np.mean(ypoints)}, std y: {np.std(ypoints)}")

        headings_same, xPos_same, yPos_same, targetXPos_same, targetYPos_same, uArray_same = sim_ra.simulate_ring_attractor(N,L,T,10,nagents,allocentricFlag,periodicflag,
                                                                                                            rEgo,rEgoTarget,Egonumber,distf,adistf,J,beta,h0s_same,h_b,dt,
                                                                                                            v0,v0t,sigma,hColl,rColl,[50],[50],xpoints,ypoints,True,False)
        print(f"final x low sigma: {xPos_same[0][-1]}, final y same: {yPos_same[0][-1]}")
        plt.figure(2)
        plt.imshow(uArray_same[:,0,:],aspect='auto')
        
        headings, xPos, yPos, targetXPos, targetYPos, uArray = sim_ra.simulate_ring_attractor(N,L,T,10,nagents,allocentricFlag,periodicflag,
                                                                                            rEgo,rEgoTarget,Egonumber,distf,adistf,J,beta,h0s_same,h_b,dt,
                                                                                            v0,v0t,sigma*2,hColl,rColl,[50],[50],xpoints,ypoints,True,False)
        print(f"final x high sigma: {xPos[0][-1]}, final y: {yPos[0][-1]}")
        plt.figure(3)
        plt.imshow(uArray[:,0,:],aspect='auto')
        
        plt.show()

def run_sims(base,change,change_var,sample_size,traj,activity,ncol,nrow):
    target_list, time_list, activity_list, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,traj,activity)
    phases = []
    angles = []
    bif_indices = []
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
estimated_sigma = helpers.get_estimated_sigma([initialxt[0],initialyt[0]],[initialxt[1],initialyt[1]],[initialxt[2],initialyt[2]],0.225,0.207,0.207,100)
h0_list = [[0.23,0.23,0.23],[0.26,0.26,0.26],[0.29,0.29,0.29],[0.32,0.32,0.32],[0.35,0.35,0.35],[0.38,0.38,0.38]]
sigma_list = [0.2,0.2,0.2,0.2,0.2,0.2]
change = {'sigma':sigma_list, 'h0':h0_list}


#run_sims(base,change,'h0',sample_size,plot_trajs,plot_neurons,3,2)
sim_random_points()
plt.show()