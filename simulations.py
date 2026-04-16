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

allocentricFlag = 1 # 1 is allo, 0 is ego
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

plot_neurons = True
plot_trajs = [True, 'scatter']
plot_sweep = not (plot_neurons or plot_trajs[0])
h0_for_plot = [0.43] #0.12 to 0.4 for even case
h0_for_sim = []
for item in h0_for_plot:
    h0_for_sim.append([item,item+0.01])
change = {'h0':h0_for_sim}
uneven = h0_for_sim[0][0]!=h0_for_sim[0][1]

sample_size = 25

success_list, target_list, time_list, inhib_list, inhib_times_list, sum_activity_list, activity_df, x_list, y_list  = repeated_sims.sample_sims(base,change,sample_size,include_trajs=plot_trajs,include_activity=plot_neurons)
p_success, se_success, p_correct, se_correct = sim_met.get_success_rate(target_list, sample_size, uneven, 1)
mean_inhib = sim_met.get_inhibition_rates(inhib_list, sample_size)

figure_num = 1

# SWEEP PLOTS

if plot_sweep:
    x_label = "h0"
    dist_y_label = "Average distance to target"
    dist_title = f" Distance to target over last quarter of simulation, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(success_list,h0_for_plot,dist_title,x_label,dist_y_label,False)
    figure_num += 1

    dist_agg_title = f"Average distance to target over last quarter of simulation, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(success_list,h0_for_plot,dist_agg_title,x_label,dist_y_label,True)
    figure_num += 1

    time_y_label = "Average time to target"
    time_title = f"Time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction, stopping distance = 0.5"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(time_list,h0_for_plot,time_title,x_label,time_y_label,False)
    figure_num += 1

    time_agg_title = f"Average time to target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction, stopping distance = 0.5"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    sim_met.plot_metric(time_list,h0_for_plot,time_agg_title,x_label,time_y_label,True)
    figure_num += 1

    p_success_y = "Probability of reaching a target"
    p_success_title = f"Probability of getting within 0.5 units of a target, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
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
    figure_num += 1

    
    n_inhib_y = "Average number of steps where every neuron was inhibited"
    n_inhib_title = f"Average number of steps where every neuron was inhibited, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction"
    plt.figure(figsize=(10,5))
    plt.figure(figure_num)
    plt.plot(h0_for_plot,mean_inhib,color='red')
    plt.xlabel(x_label)
    plt.ylabel(n_inhib_y)
    plt.title(n_inhib_title)
    figure_num += 1

# SINGLE SETTING PLOTS

if plot_neurons:   
    plt.figure(figure_num)
    mean_list = sim_met.plot_sum_activity(sum_activity_list)
    plt.title(f"Sum of all neuron activity over time, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    plt.xlabel("Time")
    plt.ylabel("Sum of neuron activity")
    figure_num+=1

    critical_points = sim_met.get_peaks(mean_list)
    # figure out a good way to get ranges as function of time (look at overall plots maybe...)

    # now instead of getting the most active neurons on average over EVERYTHING
    # just get most active in the moving forward zone (0 (?) to first critical point (min))
    # most active from min to max (if it exists)
    # if max exists get from max to end (stable zone?) 

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,True,5,0,critical_points[0],100)
    plt.title(f"Average activity of five most active neurons up to time {critical_points[0]}, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    plt.figure(figure_num)
    sim_met.plot_neurons(activity_df,False,5,0,critical_points[0],100)
    plt.title(f"Activity of five most active neurons up to time {critical_points[0]}, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1

    if len(critical_points) > 1:
        plt.figure(figure_num)
        sim_met.plot_neurons(activity_df,True,5,critical_points[0],critical_points[1],100)
        plt.title(f"Average activity of five most active neurons from time {critical_points[0]} to time {critical_points[1]}, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
        figure_num += 1

        plt.figure(figure_num)
        sim_met.plot_neurons(activity_df,False,5,critical_points[0],critical_points[1],100)
        plt.title(f"Activity of five most active neurons from time {critical_points[0]} to time {critical_points[1]}, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
        figure_num += 1

        plt.figure(figure_num)
        sim_met.plot_neurons(activity_df,True,5,critical_points[1]+200,0,100)
        plt.title(f"Average activity of five most active neurons from time {critical_points[1]+200} to end, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
        figure_num += 1

        plt.figure(figure_num)
        sim_met.plot_neurons(activity_df,False,5,critical_points[1]+200,0,100)
        plt.title(f"Activity of five most active neurons from time {critical_points[1]+200} to end, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
        figure_num += 1

    else: 
        plt.figure(figure_num)
        sim_met.plot_neurons(activity_df,True,5,critical_points[0]+200,0,100)
        plt.title(f"Average activity of five most active neurons from time {critical_points[0]+200} to end, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
        figure_num += 1

        plt.figure(figure_num)
        sim_met.plot_neurons(activity_df,False,5,critical_points[0]+200,0,100)
        plt.title(f"Activity of five most active neurons from time {critical_points[0]+200} to end, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
        figure_num += 1

if plot_trajs[0]:
    plt.figure(figsize=(7,7))
    plt.figure(figure_num)
    if plot_trajs[1] == 'heat':
        img = sim_met.plot_from_density(x_list, y_list, 30)
        plt.imshow(img)
    if plot_trajs[1] == 'scatter':
        sim_met.plot_traj(x_list,y_list,initialxt,initialyt,sample_size)
    plt.title(f"Trajectory plot, h0: {h0_for_sim[0]}, {"allocentric" if allocentricFlag==1 else "egocentric"}, {"uneven" if uneven else "even"} attraction")
    figure_num += 1


plt.show()