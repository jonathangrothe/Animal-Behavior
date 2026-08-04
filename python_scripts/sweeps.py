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
sigma = 0.5
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

include_pos = True
include_neurons = True
sample_size = 5
sigma_start = 0.05
sigma_finish = 0.75
n_sigma = 30
base_sigma = np.linspace(sigma_start,sigma_finish,num=n_sigma)
h0_start = 0.2
h0_finish = 0.4
n_h0 = 30
h0_range= np.linspace(h0_start,h0_finish,num=n_h0)
print(f"sigma diff: {base_sigma[1]-base_sigma[0]}, h0 diff: {h0_range[1]-h0_range[0]}")
h0_list = []
for item in h0_range:
    h0_list.append([item,item,item])

target_grid = []
p_targets_grid = []
phase_grid = []
time_grid = []
angle_grid = []
angle_2_grid = []
angle_len_grid = []

n_thresh = 10
dir_thresh_a1_arr = np.zeros((n_sigma*n_thresh,n_h0))
dir_thresh_a2_arr = np.zeros((n_sigma*n_thresh,n_h0))
dir_thresh_len_arr = np.zeros((n_sigma*n_thresh,n_h0))

delta_thresh_a1_arr = np.zeros((n_sigma*n_thresh,n_h0))
delta_thresh_a2_arr = np.zeros((n_sigma*n_thresh,n_h0))
delta_thresh_len_arr = np.zeros((n_sigma*n_thresh,n_h0))

direction_thresholds = np.linspace(0.01,np.pi/15,num=n_thresh)
delta_thresholds = np.linspace(np.pi/1000,np.pi/300,num=n_thresh)



for sigma_index in range(n_sigma):
    sigma_list = [base_sigma[sigma_index]]*n_h0
    change= {'h0':h0_list,
             'sigma':sigma_list}
    sample_time = time.perf_counter()
    target_list, time_list, activity_list, x_list, y_list, headings_list  = repeated_sims.sample_sims(base,change,sample_size,include_pos,include_neurons)
    end_sample_time = time.perf_counter()
    sampling_time = end_sample_time - sample_time
    print(f"Sampling time: {sampling_time:.6f} seconds")
    analysis_time = time.perf_counter()
    p_target, se_target = sim_met.get_success_rate(target_list, sample_size, ntargets)
    p_targets_grid.append(p_target)
    target_reached = [0 if (x == 0) or (x == 2) else x for x in target_list]
    total_sims = len(target_list)
    phases = []
    angles = []
    angles2 = []
    angle_len = []
    dir_angle1_arr = np.zeros((n_thresh,total_sims))
    dir_angle2_arr = np.zeros((n_thresh,total_sims))
    dir_anglelen_arr = np.zeros((n_thresh,total_sims))
    delta_angle1_arr = np.zeros((n_thresh,total_sims))
    delta_angle2_arr = np.zeros((n_thresh,total_sims))
    delta_anglelen_arr = np.zeros((n_thresh,total_sims))

    for i in range(total_sims):
        curr_activity = activity_list[i]
        curr_headings = headings_list[i][0]
        curr_xpos = x_list[i:(i+1)][0]
        curr_ypos = y_list[i:(i+1)][0]
        phase = sim_met.get_bump_type(curr_activity)
        phases.append(phase)
        if target_list[i] != -1:
            sim_indices, sim_angle = sim_met.get_bifurcation_angle(curr_xpos,curr_ypos,curr_headings)
            sim_len = len(sim_angle)
            if sim_len > 5:
                sim_len = 5
            angle_len.append(sim_len)
            if sim_len > 0:
                angles.append(sim_angle[0])
            else:
                angle_len.append(0)
                angles.append(np.pi)
            if sim_len > 1:
                angles2.append(sim_angle[1])
            else:
                angles2.append(np.pi)
        else:
            angle_len.append(0)
            angles.append(np.pi)
            angles2.append(np.pi)

        # for each distance threshold get the angles and len IF IT REACHED A TARGET
        if target_list[i] != -1:
            for indexdir, dirthresh in enumerate(direction_thresholds):
                dir_thresh_indices, dir_thresh_angles = sim_met.get_bifurcation_angle(curr_xpos, curr_ypos,curr_headings,dirthresh)
                dir_thresh_len = len(dir_thresh_angles)
                if dir_thresh_len > 5: 
                    dir_thresh_len = 5
                if dir_thresh_len > 0:
                    dir_angle1_arr[indexdir,i] = dir_thresh_angles[0]
                    dir_anglelen_arr[indexdir,i] = dir_thresh_len
                else:
                    dir_anglelen_arr[indexdir,i] = 0
                    dir_angle1_arr[indexdir,i] = np.pi
                if dir_thresh_len > 1:
                    dir_angle2_arr[indexdir,i] = dir_thresh_angles[1]
                else:
                    dir_angle2_arr[indexdir,i] = np.pi

            for indexdelta, deltathresh in enumerate(delta_thresholds):
                delta_thresh_indices, delta_thresh_angles = sim_met.get_bifurcation_angle(curr_xpos, curr_ypos,curr_headings,deltathresh)
                delta_thresh_len = len(delta_thresh_angles)
                if delta_thresh_len > 5:
                    delta_thresh_len = 5
                if delta_thresh_len > 0:
                    delta_angle1_arr[indexdelta,i] = delta_thresh_angles[0]
                    delta_anglelen_arr[indexdelta,i] = delta_thresh_len
                else:
                    delta_anglelen_arr[indexdelta,i] = 0
                    delta_angle1_arr[indexdelta,i] = np.pi
                if delta_thresh_len > 1:
                    delta_angle2_arr[indexdelta,i] = delta_thresh_angles[1]
                else:
                    delta_angle2_arr[indexdelta,i] = np.pi
        else:
            dir_anglelen_arr[:,i] = 0
            dir_angle1_arr[:,i] = np.pi
            dir_angle2_arr[:,i] = np.pi
            delta_anglelen_arr[:,i] = 0
            delta_angle1_arr[:,i] = np.pi
            delta_angle2_arr[:,i] = np.pi


    grid_phases = []
    grid_targets = []
    grid_times = []
    grid_angles = []
    grid_angles2 = []
    grid_angle_len = []
    for s in range(n_h0):
        sim_phases = phases[s*sample_size:(s+1)*sample_size]
        sim_targets = target_reached[s*sample_size:(s+1)*sample_size]
        sim_times = time_list[s*sample_size:(s+1)*sample_size]
        sim_angles = angles[s*sample_size:(s+1)*sample_size]
        sim_angles2 = angles2[s*sample_size:(s+1)*sample_size]
        sim_anglelen = angle_len[s*sample_size:(s+1)*sample_size]
        n0 = sim_phases.count(0)
        n1 = sim_phases.count(1)
        n2 = sim_phases.count(2)
        n3 = sim_phases.count(3)
        nOther = sim_phases.count(4)
        nreach = sim_targets.count(1) + sim_targets.count(0)
        reach = 1
        if 0 < nreach < sample_size:
            reach = 0 
        elif nreach == 0:
            reach = -1
        for m in range(n_thresh):
            dir1_sim = dir_angle1_arr[m,s*sample_size:(s+1)*sample_size]
            dir2_sim = dir_angle2_arr[m,s*sample_size:(s+1)*sample_size]
            dirlen_sim = dir_anglelen_arr[m,s*sample_size:(s+1)*sample_size]
            dir_thresh_a1_arr[m*n_sigma+sigma_index,s] = np.mean(dir1_sim)
            dir_thresh_a2_arr[m*n_sigma+sigma_index,s] = np.mean(dir2_sim)
            dir_thresh_len_arr[m*n_sigma+sigma_index,s] = np.mean(dirlen_sim)

            for p in range(n_thresh):
                delta1_sim = delta_angle1_arr[p,s*sample_size:(s+1)*sample_size]
                delta2_sim = delta_angle2_arr[p,s*sample_size:(s+1)*sample_size]
                deltalen_sim = delta_anglelen_arr[p,s*sample_size:(s+1)*sample_size]
                delta_thresh_a1_arr[p*n_sigma+sigma_index,s] = np.mean(delta1_sim)
                delta_thresh_a2_arr[p*n_sigma+sigma_index,s] = np.mean(delta2_sim)
                delta_thresh_len_arr[p*n_sigma+sigma_index,s] = np.mean(deltalen_sim)


        grid_phases.append(np.argmax([n0,n1,n2,n3,nOther]))
        grid_targets.append(reach)
        grid_times.append(np.mean(sim_times))
        grid_angles.append(np.mean(sim_angles))
        grid_angles2.append(np.mean(sim_angles2))
        grid_angle_len.append(np.mean(sim_anglelen))


    phase_grid.append(grid_phases)
    target_grid.append(grid_targets)
    time_grid.append(grid_times)
    angle_grid.append(grid_angles)
    angle_2_grid.append(grid_angles2)
    angle_len_grid.append(grid_angle_len)


    end_analysis_time = time.perf_counter()
    analyzing_time = end_analysis_time - analysis_time
    print(f"Analysis time: {analyzing_time:.6f} seconds")


target_df = pd.DataFrame(target_grid,columns=h0_range,index=base_sigma)
phase_df = pd.DataFrame(phase_grid,columns=h0_range,index=base_sigma)
time_df = pd.DataFrame(time_grid,columns=h0_range,index=base_sigma)
angle_df = pd.DataFrame(angle_grid,columns=h0_range,index=base_sigma)
angle_2_df = pd.DataFrame(angle_2_grid,columns=h0_range,index=base_sigma)
angle_len_df = pd.DataFrame(angle_len_grid,columns=h0_range,index=base_sigma)

p0_arr = np.zeros((n_sigma,n_h0))
p1_arr = np.zeros((n_sigma,n_h0))
p2_arr = np.zeros((n_sigma,n_h0))
for r, row in enumerate(p_targets_grid):
    for c, entry in enumerate(row):
        p0_arr[r][c] = entry[0]
        p1_arr[r][c] = entry[1]
        p2_arr[r][c] = entry[2]

print(f"p0 array: {p0_arr}")
print(f"p1 array: {p1_arr}")
print(f"p2 array: {p2_arr}")



subfigs_metric, axs_metric = plt.subplots(nrows=2,ncols=3, figsize = (17.25,8),num=1)
axs_metric = axs_metric.flatten()

bwr_r = plt.colormaps['bwr_r']
discrete_bwrr = bwr_r.resampled(3)(np.linspace(0,1,3))
plasma = plt.colormaps['plasma']
discrete_plasma = plasma.resampled(4)(np.linspace(0,1,4))
cmap1 = mcolors.ListedColormap(discrete_bwrr)
cmap2 = mcolors.ListedColormap(discrete_plasma)
cmap3 = 'inferno_r'
cmap4 = 'viridis'
cmap5 = 'magma'
boundaries_tar = np.arange(-1,3) - 0.5
norm_tar = mcolors.BoundaryNorm(boundaries_tar,cmap1.N)
boundaries_phase = np.arange(5) - 0.5
norm_phase = mcolors.BoundaryNorm(boundaries_phase,cmap2.N)
categories_tar = ['fails', 'both', 'reaches']
categories_phase = ['0 bumps', '1 bump', '2 bumps','3 bumps']

im1 = axs_metric[0].imshow(target_df, cmap=cmap1,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im2 = axs_metric[1].imshow(phase_df,cmap=cmap2,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im3 = axs_metric[2].imshow(time_df,cmap=cmap3,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im4 = axs_metric[3].imshow(angle_df,cmap=cmap4,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
shared_norm = im4.norm 
im5 = axs_metric[4].imshow(angle_2_df,cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im6 = axs_metric[5].imshow(angle_len_df,cmap=cmap5,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
shared_len_norm = im6.norm
cbar1 = plt.colorbar(im1, ticks=np.arange(-1,2))
cbar1.ax.set_yticklabels(categories_tar)
cbar2 = plt.colorbar(im2, ticks=np.arange(4))
cbar2.ax.set_yticklabels(categories_phase)
cbar3 = plt.colorbar(im3)
cbar3.set_label('Time to target')
cbar4 = plt.colorbar(im4)
cbar4.set_label('angle of first bifurcation')
cbar5 = plt.colorbar(im5)
cbar5.set_label('angle of second bifurcation')
cbar6 = plt.colorbar(im6)
cbar6.set_label('number of bifurcations')

subfigs_metric.suptitle(f"Heatmaps for 2π/3 between targets, average of {sample_size} samples")
width_ratio = [1]*n_thresh
width_ratio.append(0.08)
subfigs_dir, axs_dir = plt.subplots(nrows=3,ncols=n_thresh+1,figsize=(18,6),gridspec_kw={'width_ratios': width_ratio},num=2)
subfigs_delta, axs_delta = plt.subplots(nrows=3,ncols=n_thresh+1,figsize=(18,6),gridspec_kw={'width_ratios': width_ratio},num=3)
axs_dir = axs_dir.flatten()
axs_delta = axs_delta.flatten()

for t in range(n_thresh):
    axs_dir[t].imshow(dir_thresh_a1_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_dir[t+n_thresh+1].imshow(dir_thresh_a2_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_dir[t+2*(n_thresh+1)].imshow(dir_thresh_len_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap5,norm=shared_len_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_dir[t].set_xticks([])
    axs_dir[t].set_yticks([])
    axs_dir[t+n_thresh+1].set_xticks([])
    axs_dir[t+n_thresh+1].set_yticks([])
    axs_dir[t+2*(n_thresh+1)].set_xticks([])
    axs_dir[t+2*(n_thresh+1)].set_yticks([])
    axs_dir[t].set_title(f"{direction_thresholds[t]:.4f}")

    axs_delta[t].imshow(delta_thresh_a1_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_delta[t+n_thresh+1].imshow(delta_thresh_a2_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_delta[t+2*(n_thresh+1)].imshow(delta_thresh_len_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap5,norm=shared_len_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_delta[t].set_xticks([])
    axs_delta[t].set_yticks([])
    axs_delta[t+n_thresh+1].set_xticks([])
    axs_delta[t+n_thresh+1].set_yticks([])
    axs_delta[t+2*(n_thresh+1)].set_xticks([])
    axs_delta[t+2*(n_thresh+1)].set_yticks([])
    axs_delta[t].set_title(f"threshold: {delta_thresholds[t]:.4f}")

cbar_dir1 = subfigs_dir.colorbar(axs_dir[n_thresh-1].images[0], cax=axs_dir[n_thresh])
cbar_dir1.set_label('angle of first bifurcation')
cbar_dir2 = subfigs_dir.colorbar(axs_dir[2*n_thresh].images[0], cax=axs_dir[2*n_thresh+1])
cbar_dir2.set_label('angle of second bifurcation')
cbar_dirlen = subfigs_dir.colorbar(axs_dir[3*n_thresh+1].images[0], cax=axs_dir[3*n_thresh+2])
cbar_dirlen.set_label('number of bifurcations')

cbar_delta1 = subfigs_delta.colorbar(axs_delta[n_thresh-1].images[0], cax=axs_delta[n_thresh])
cbar_delta1.set_label('angle of first bifurcation')
cbar_delta2 = subfigs_delta.colorbar(axs_delta[2*n_thresh].images[0], cax=axs_delta[2*n_thresh+1])
cbar_delta2.set_label('angle of second bifurcation')
cbar_deltalen = subfigs_delta.colorbar(axs_delta[3*n_thresh+1].images[0], cax=axs_delta[3*n_thresh+2])
cbar_deltalen.set_label('number of bifurcations')

subfigs_dir.suptitle("Direction threshold analysis for angle 1, angle 2, and number of bifurcations")
subfigs_delta.suptitle("Delta threshold analysis for angle 1, angle 2, and number of bifurcations")

subfigs_pr, axs_pr = plt.subplots(nrows=1,ncols=3,figsize=(17.25,8),num=4)
p_t0 = axs_pr[0].imshow(p0_arr, cmap = bwr_r, origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
p_t1 = axs_pr[1].imshow(p1_arr, cmap = bwr_r, origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
p_t2 = axs_pr[2].imshow(p2_arr, cmap = bwr_r, origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

#print(target_df)
#print(phase_df)
#print(time_df)
#print(angle_df)
#print(angle_2_df)
#print(angle_len_df)

#print(dir_anglelen_arr)
#print(delta_anglelen_arr)

subfigs_metric.tight_layout()
subfigs_dir.tight_layout()
subfigs_delta.tight_layout()
plt.show()
