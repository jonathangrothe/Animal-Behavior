import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
h0s = [0.25,0.25,0.25]
h_b = 0.2
sigma = 0.2
beta = 100

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

sample_size = 15
h0_base = 0.275
num_changes = 50
changing_h0 = np.linspace(0.03,0.05,num=num_changes)
h0_list_0 = []
for item in changing_h0:
    h0_list_0.append([item+h0_base,h0_base,h0_base])
sigma_list = np.linspace(0.08,0.12,num=16)
subfigs_pr, axs_pr = plt.subplots(nrows=4,ncols=4,figsize=(16,12),num=1)
axs_pr = axs_pr.flatten()
geometry_list = [[[35,50,65],[50+15*np.sqrt(3),80,50+15*np.sqrt(3)]]]
for geo_list in geometry_list:
    for ind, sigma in enumerate(sigma_list):
        print(f"NEW SIGMA, iteration: {ind}, sigma: {sigma}")
        #geom = 2*np.pi/3 - ind*np.pi/6
        base['initialxt'] = geo_list[0]
        base['initialyt'] = geo_list[1]
        base['sigma'] = sigma
        change = {'h0':h0_list_0}
        sample_start = time.perf_counter()
        target_list, time_list, activity_list, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,True,False)
        sample_end = time.perf_counter()
        print(f"sample time: {sample_end-sample_start}")
        p_target, se_target = sim_met.get_success_rate(target_list, sample_size, ntargets)
        p0_arr = np.zeros(num_changes)
        p1_arr = np.zeros(num_changes)
        p2_arr = np.zeros(num_changes)
        for r, entry in enumerate(p_target):
                p0_arr[r] = entry[0]
                p1_arr[r] = entry[1]
                p2_arr[r] = entry[2]

        axs_pr[ind].plot(changing_h0,p0_arr,color="#e72945",label= 'left')
        axs_pr[ind].plot(changing_h0,p1_arr,color="#1c12b4", label='center')
        axs_pr[ind].plot(changing_h0,p2_arr,color="#a7b0ae", label='right')
        axs_pr[ind].set_title(f"sigma: {sigma:.4f}")
        #prob_df = pd.DataFrame({'left':p0_arr,'center':p1_arr,'right':p2_arr})
        #prob_df.to_csv(f'016sigma02125h0_{ind}.csv', index=False)

subfigs_pr.suptitle(f"probability of reaching target (left is getting better), angle: {np.pi/6:.4f}, base h0: {h0_base}")
subfigs_pr.supylabel("probability of reaching each target")
axs_pr[ind].legend()
subfigs_pr.supxlabel("difference in h0 from left target to others")

plt.show()