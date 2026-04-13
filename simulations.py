import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor as sim_ra
import simulation_metrics as sim_met
import repeated_sims

# --------  PARAMETERS --------

L = 100 # width of grid
ntargets = 2
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 20
    initialy[a] = 50
initialxt = [80,80]
initialyt = [20,80]

# number of time steps
T = 5000
periodicflag = 0
rEgo = 0 # radius to switch to egocentric (for when close to another agent)
rEgoTarget = 0 # radius for switch near target 
Egonumber = 1

# collision avoidance:
hColl = -10
rColl = 0

# signal decay
distf = 0
adistf = 1

dt = 0.1
v0 = 0.05

v0t = np.zeros(ntargets)
for i in range(ntargets):
    v0t[i] = 0

N = 100 # number of neurons
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

allocentricFlag = 0 # 1 is allo, 0 is ego
h0s = [0.25,0.25] # attraction vector, first are attraction for targets, then agents
h_b = 0.2
sigma = 0.5
beta = 100


# -------- Running the simulation --------

# first we set out base parameters using the initializations above

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

uneven = True # flag for whether or not the attractions are even
plot_neurons = False
plot_trajs = False
h0_for_plot = np.linspace(0.2,0.29,num=15) #0.12 to 0.4 for even case
h0_for_sim = []
for item in h0_for_plot:
    h0_for_sim.append([item,item+0.01])
change = {'h0':h0_for_sim}

sample_size = 10

success_list, target_list, time_list, inhib_list, inhib_times_list, activity_df, img  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
p_success, se_success, p_correct, se_correct = sim_met.get_success_rate(target_list, sample_size, True, 1)
mean_inhib = sim_met.get_inhibition_rates(inhib_list, sample_size)

if plot_neurons == True:
    plt.figure(2)
    sim_met.plot_neurons(activity_df, 0.75, tstart=312,tstop=332)
    plt.title("Activity of most active neurons (0.75 as active as most active neuron) from time step 300 to time step 350")
    plt.show()

# SAMPLING PLOTS

x_label = "h0"
dist_y_label = "Average distance to target"
dist_title = "Average distance to target over last quarter of simulation, egocentric, uneven attraction"
plt.figure(figsize=(10,5))
plt.figure(2)
sim_met.plot_metric(success_list,h0_for_plot,dist_title,x_label,dist_y_label)
plt.show(block=False)

time_y_label = "Average time to target"
time_title = "Time to target, egocentric, uneven attraction, stopping distance = 0.5"
plt.figure(figsize=(10,5))
plt.figure(3)
sim_met.plot_metric(time_list,h0_for_plot,time_title,x_label,time_y_label)
plt.show(block=False)

p_success_y = "Probability of reaching a target"
p_success_title = "Probability of getting within 0.5 units of a target, egocentric, uneven attraction"
plt.figure(figsize=(10,5))
plt.figure(4)
if uneven: 
    plt.plot(h0_for_plot,p_correct,color='Green', label="Correct target")
    plt.plot(h0_for_plot, np.add(p_correct,se_correct), color = 'green', label = 'standard error for correct', linestyle = ':')
    plt.plot(h0_for_plot, np.subtract(p_correct,se_correct), color = 'green', linestyle = ':')
    plt.plot(h0_for_plot,p_success,color='Blue', label="Any target")
    plt.plot(h0_for_plot, np.add(p_success,se_success), color = 'blue', label = 'standard error for any target', linestyle = ':')
    plt.plot(h0_for_plot, np.subtract(p_success,se_success), color = 'blue', linestyle = ':')
    plt.legend()
if not uneven:
    plt.plot(h0_for_plot,p_success,color='Blue')
    plt.plot(h0_for_plot, np.add(p_success,se_success), color = 'blue', label = 'standard error', linestyle = ':')
    plt.plot(h0_for_plot, np.subtract(p_success,se_success), color = 'blue', linestyle = ':')
plt.xlabel(x_label)
plt.ylabel(p_success_y)
plt.title(p_success_title)
plt.show(block=False)

plt.figure(5)
n_inhib_y = "Average number of steps where every neuron was inhibited"
n_inhib_title = "Average number of steps where every neuron was inhibited, egocentric, uneven attraction"
plt.figure(figsize=(10,5))
plt.plot(h0_for_plot,mean_inhib,color='red')
plt.xlabel(x_label)
plt.ylabel(n_inhib_y)
plt.title(n_inhib_title)
plt.show()



#TRAJECTORY PLOTS - NEEDS REWORK
if plot_trajs == True:
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()

'''
plt.figure(1)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Mean trajectories for allocentric, even attraction. h0:0.68(blue)->0.8(red)',5, 'mean')

plt.figure(2)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Min trajectories for allocentric, even attraction. h0:0.68(blue)->0.8(red)',5, 'min')

plt.figure(3)
simulation_metrics.plot_trajectories(x_trajs,y_trajs,initialxt,initialyt,
                                     plt.cm.coolwarm(np.linspace(0, 1, 5)),L,
                                     'Max trajectories for allocentric, even attraction. h0:0.68(blue)->0.8(red)',5, 'max')
plt.show(block=False)
'''
