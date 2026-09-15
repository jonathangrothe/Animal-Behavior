'''
A file for running a large number of simulations in a sweep over a parameter space
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from . import simulation_metrics as sim_met
from . import repeated_sims

start_time = time.perf_counter()

L = 100
ntargets = 3
nagents = 1
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 50
    initialy[a] = 20
initialxt = [50-15*np.sqrt(3),50,50+15*np.sqrt(3)]
initialyt = [35,50,35]

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
h0s = [0.23,0.23]
h_b = 0.2
sigma = 0.4
beta = 100

# -------- Running the simulation --------
u0 = np.zeros((N,1))
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
        'u0':u0,
        'factor':0.2,
        'seed':False}

# defining what we are sweeping over

include_pos = True
include_neurons = True
sample_size = 30
'''
h0_start = 0.15
h0_finish = 0.4
n_h0 = 100
sigma_start = 0.1
sigma_finish = 1
n_sigma = 100
base_h0 = np.linspace(h0_start,h0_finish,num=n_h0)
base_sigma = np.linspace(sigma_start,sigma_finish,num=n_sigma)
h0_list = []
sigma_list = []
for s in range(n_sigma):
    for h in range(n_h0):
        h0_list.append([base_h0[h],base_h0[h]])
        sigma_list.append(base_sigma[s])
'''
h0_diff_start = 0
h0_diff_finish = 0.06
num_h0_diff = 75
h0_diff = np.linspace(h0_diff_start,h0_diff_finish,num=num_h0_diff)
h0_diff_list = []
base_h0_val = 0.25
for d in h0_diff:
    h0_diff_list.append([base_h0_val,base_h0_val,base_h0_val+d])

fig_init = plt.figure(layout='constrained',figsize = (10,5),num=1)
subfigs_init = fig_init.subfigures(1,1,squeeze=False)
axs = subfigs_init[0][0].subplots(1,1,squeeze=False)

change= {'h0':h0_diff_list}
factor_list = [0.2,2]

for f in range(1):
    base['factor'] = factor_list[f]
    sample_time = time.perf_counter()
    target_list, time_list, activity_list, x_list, y_list, headings_list, init_heading  = repeated_sims.sample_sims(base,change,sample_size,include_pos,include_neurons)
    end_sample_time = time.perf_counter()
    sampling_time = end_sample_time - sample_time
    print(f"Sampling time: {sampling_time:.6f} seconds")

    analysis_time = time.perf_counter()

    success_rate, success_se = sim_met.get_success_rate(target_list,sample_size,ntargets)
    print(success_rate)
    print(success_se)
    probs = []
    low_se = []
    high_se = []
    prob_mid = []
    low_se_mid = []
    high_se_mid = []
    prob_wrong = []
    low_se_wrong = []
    high_se_wrong = []
    for p,se in zip(success_rate,success_se):
        print(f"p: {p}, se: {se}")
        probs.append(p[2])
        low_se.append(p[2]-se[2])
        high_se.append(min(1,p[2]+se[2]))
        prob_mid.append(p[1])
        low_se_mid.append(p[1]-se[1])
        high_se_mid.append(min(1,p[1]+se[1]))
        prob_wrong.append(p[0])
        low_se_wrong.append(p[0]-se[0])
        high_se_wrong.append(min(1,p[0]+se[0]))
    '''
    targ_reached = []
    time_reached = []
    for setting in range(n_h0*n_sigma):
        time_sample = time_list[setting*sample_size:(setting+1)*sample_size]
        target_sample = target_list[setting*sample_size:(setting+1)*sample_size]
        n_fail = target_sample.count(-1)
        n_succ = sample_size - n_fail
        time_reached.append(np.mean(time_sample))
        targ_reached.append(n_succ/sample_size)
    targ_grid = np.zeros((n_h0,n_sigma))
    time_grid = np.zeros((n_h0,n_sigma))
    for p in range(n_sigma):
        targ_row = targ_reached[p*n_sigma:(p+1)*n_sigma]
        targ_grid[p,:] = targ_row
        time_row = time_reached[p*n_sigma:(p+1)*n_sigma]
        time_grid[p,:] = time_row

    bwr_r = plt.colormaps['bwr_r']
    plasma_r = plt.colormaps['plasma_r']
    axs0 = subfigs_init[0][0].subplots(1,2)
    axs0[0].imshow(targ_grid,cmap=bwr_r,origin='lower',aspect='auto',extent=[base_h0[0], base_h0[-1], base_sigma[0], base_sigma[-1]])
    axs0[0].set_xlabel("h0")
    axs0[0].set_ylabel("sigma")
    axs0[0].text(-0.1,1.05,'A',transform=axs0[0].transAxes,size=14,weight="bold")

    axs0[1].imshow(time_grid,cmap=plasma_r,origin='lower',aspect='auto',extent=[base_h0[0], base_h0[-1], base_sigma[0], base_sigma[-1]])
    axs0[1].set_xlabel("h0")
    axs0[1].set_ylabel("sigma")
    axs0[1].text(-0.1,1.05,'B',transform=axs0[1].transAxes,size=14,weight="bold")
    '''

    # plotting the output
    
    axs[0][0].plot(h0_diff,probs,c='blue')
    axs[0][0].plot(h0_diff,low_se,c='red',ls='--',alpha=0.4)
    axs[0][0].plot(h0_diff,high_se,c='red',ls='--',alpha=0.4)
    axs[0][0].plot(h0_diff,prob_mid, c='green')
    axs[0][0].plot(h0_diff,low_se_mid,c='red',ls='--',alpha=0.4)
    axs[0][0].plot(h0_diff,high_se_mid,c='red',ls='--',alpha=0.4)
    axs[0][0].plot(h0_diff,prob_wrong, c='yellow')
    axs[0][0].plot(h0_diff,low_se_wrong,c='red',ls='--',alpha=0.4)
    axs[0][0].plot(h0_diff,high_se_wrong,c='red',ls='--',alpha=0.4)
    end_analysis_time = time.perf_counter()
    analyzing_time = end_analysis_time - analysis_time
    print(f"Analysis time: {analyzing_time:.6f} seconds")
#axs[0].text(-0.1,1.05,'A',transform=axs[0].transAxes,size=14,weight="bold")
#axs[1].text(-0.1,1.05,'B',transform=axs[1].transAxes,size=14,weight="bold")
#axs[2].text(-0.1,1.05,'C',transform=axs[2].transAxes,size=14,weight="bold")
#axs[3].text(-0.1,1.05,'D',transform=axs[3].transAxes,size=14,weight="bold")

plt.show()
