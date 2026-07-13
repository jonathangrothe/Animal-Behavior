import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import simulation_metrics as sim_met
import repeated_sims


start_time = time.perf_counter()

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
sample_size = 10
sigma_start = 0.14
sigma_finish = 0.26
n_sigma = 49
base_sigma = np.linspace(sigma_start,sigma_finish,num=n_sigma)
h0_start = 0.19
h0_finish = 0.24
n_h0 = 101
h0_range= np.linspace(h0_start,h0_finish,num=n_h0)
print(f"sigma diff: {base_sigma[1]-base_sigma[0]}, h0 diff: {h0_range[1]-h0_range[0]}")
h0_list = []
for item in h0_range:
    h0_list.append([item,item,item])

target_grid = []
phase_grid = []
time_grid = []
angle_grid = []
angle_2_grid = []
angle_len_grid = []

n_thresh = 8
act_thresh_a1_arr = np.zeros((n_sigma*n_thresh,n_h0))
act_thresh_a2_arr = np.zeros((n_sigma*n_thresh,n_h0))
act_thresh_len_arr = np.zeros((n_sigma*n_thresh,n_h0))

mov_thresh_a1_arr = np.zeros((n_sigma*n_thresh,n_h0))
mov_thresh_a2_arr = np.zeros((n_sigma*n_thresh,n_h0))
mov_thresh_len_arr = np.zeros((n_sigma*n_thresh,n_h0))

act_thresholds = np.linspace(0.1,0.5,num=n_thresh)
mov_thresholds = np.linspace(1,8,num=n_thresh)
mov_thresholds_sq = mov_thresholds**2 


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
    target_reached = [0 if (x == 0) or (x == 2) else x for x in target_list]
    total_sims = len(target_list)
    phases = []
    angles = []
    angles2 = []
    angle_len = []
    act_angle1_arr = np.zeros((n_thresh,total_sims))
    act_angle2_arr = np.zeros((n_thresh,total_sims))
    act_anglelen_arr = np.zeros((n_thresh,total_sims))
    mov_angle1_arr = np.zeros((n_thresh,total_sims))
    mov_angle2_arr = np.zeros((n_thresh,total_sims))
    mov_anglelen_arr = np.zeros((n_thresh,total_sims))

    for i in range(total_sims):
        curr_activity = activity_list[i]
        curr_xpos = x_list[i:(i+1)][0]
        curr_ypos = y_list[i:(i+1)][0]
        phase = sim_met.get_bump_type(curr_activity)
        phases.append(phase)
        sim_indices, sim_angle = sim_met.get_bifurcation_angle(curr_xpos, curr_ypos,0.25,25)
        sim_len = len(sim_angle)
        if sim_len > 0:
            angles.append(sim_angle[0])
            if sim_angle[0] == np.pi and sim_len == 1:
                angle_len.append(0)
            else:
                angle_len.append(sim_len)
        else:
            angle_len.append(0)
            angles.append(np.pi)
        if sim_len > 1:
            angles2.append(sim_angle[1])
        else:
            angles2.append(np.pi)

        # for each threshold: 
        # get angle1, angle2, len(angle)
        for indexa, athresh in enumerate(act_thresholds):
            athresh_indices, athresh_angles = sim_met.get_bifurcation_angle(curr_xpos, curr_ypos,athresh,25)
            athresh_len = len(athresh_angles)
            if athresh_len > 0:
                act_angle1_arr[indexa,i] = athresh_angles[0]
                if athresh_angles[0] == np.pi and athresh_len == 1:
                    act_anglelen_arr[indexa,i] = 0
                else:
                    act_anglelen_arr[indexa,i] = athresh_len
            else:
                act_anglelen_arr[indexa,i] = 0
                act_angle1_arr[indexa,i] = np.pi
            if athresh_len > 1:
                act_angle2_arr[indexa,i] = athresh_angles[1]
            else:
                act_angle2_arr[indexa,i] = np.pi

        # for each distance threshold:
        # get angle1, angle2, len(angle)
        for indexm, mthresh in enumerate(mov_thresholds_sq):
            mthresh_indices, mthresh_angles = sim_met.get_bifurcation_angle(curr_xpos, curr_ypos,0.25,mthresh)
            mthresh_len = len(mthresh_angles)
            if mthresh_len > 0:
                mov_angle1_arr[indexm,i] = mthresh_angles[0]
                if mthresh_angles[0] == np.pi and mthresh_len == 1:
                    mov_anglelen_arr[indexm,i] = 0
                else:
                    mov_anglelen_arr[indexm,i] = mthresh_len
            else:
                mov_anglelen_arr[indexm,i] = 0
                mov_angle1_arr[indexm,i] = np.pi
            if mthresh_len > 1:
                mov_angle2_arr[indexm,i] = mthresh_angles[1]
            else:
                mov_angle2_arr[indexm,i] = np.pi

        
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
        # for each threshold:
        # access all the samples
        # aggregate all the samples
        # put them into the right spot on the grid 
        for a in range(n_thresh):
            a1_sim = act_angle1_arr[a,s*sample_size:(s+1)*sample_size]
            a2_sim = act_angle2_arr[a,s*sample_size:(s+1)*sample_size]
            alen_sim = act_anglelen_arr[a,s*sample_size:(s+1)*sample_size]
            act_thresh_a1_arr[a*n_sigma+sigma_index,s] = np.mean(a1_sim)
            act_thresh_a2_arr[a*n_sigma+sigma_index,s] = np.mean(a2_sim)
            act_thresh_len_arr[a*n_sigma+sigma_index,s] = np.mean(alen_sim)
        
        for m in range(n_thresh):
            m1_sim = mov_angle1_arr[m,s*sample_size:(s+1)*sample_size]
            m2_sim = mov_angle2_arr[m,s*sample_size:(s+1)*sample_size]
            mlen_sim = act_anglelen_arr[m,s*sample_size:(s+1)*sample_size]
            mov_thresh_a1_arr[m*n_sigma+sigma_index,s] = np.mean(m1_sim)
            mov_thresh_a2_arr[m*n_sigma+sigma_index,s] = np.mean(m2_sim)
            mov_thresh_len_arr[m*n_sigma+sigma_index,s] = np.mean(mlen_sim)


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

subfigs_metric.suptitle(f"Heatmaps for π/6 between targets, average of {sample_size} samples")
width_ratio = [1]*n_thresh
width_ratio.append(0.08)
subfigs_act, axs_act = plt.subplots(nrows=3,ncols=n_thresh+1,figsize=(18,6),gridspec_kw={'width_ratios': width_ratio},num=2)
axs_act = axs_act.flatten()
subfigs_mov, axs_mov = plt.subplots(nrows=3,ncols=n_thresh+1,figsize=(18,6),gridspec_kw={'width_ratios': width_ratio},num=3)
axs_mov = axs_mov.flatten()

for t in range(n_thresh):
    axs_act[t].imshow(act_thresh_a1_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_act[t+n_thresh+1].imshow(act_thresh_a2_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_act[t+2*(n_thresh+1)].imshow(act_thresh_len_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap5,norm=shared_len_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_mov[t].imshow(mov_thresh_a1_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_mov[t+n_thresh+1].imshow(mov_thresh_a2_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap4,norm=shared_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_mov[t+2*(n_thresh+1)].imshow(mov_thresh_len_arr[t*n_sigma:(t+1)*n_sigma,:],cmap=cmap5,norm=shared_len_norm,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
    axs_act[t].set_xticks([])
    axs_act[t].set_yticks([])
    axs_act[t+n_thresh+1].set_xticks([])
    axs_act[t+n_thresh+1].set_yticks([])
    axs_act[t+2*(n_thresh+1)].set_xticks([])
    axs_act[t+2*(n_thresh+1)].set_yticks([])
    axs_mov[t].set_xticks([])
    axs_mov[t].set_yticks([])
    axs_mov[t+n_thresh+1].set_xticks([])
    axs_mov[t+n_thresh+1].set_yticks([])
    axs_mov[t+2*(n_thresh+1)].set_xticks([])
    axs_mov[t+2*(n_thresh+1)].set_yticks([])
cbar_a1 = subfigs_act.colorbar(axs_act[n_thresh-1].images[0], cax=axs_act[n_thresh],aspect=20)
cbar_a1.set_label('angle of first bifurcation')
cbar_a2 = subfigs_act.colorbar(axs_act[2*n_thresh].images[0], cax=axs_act[2*n_thresh+1],aspect=30)
cbar_a2.set_label('angle of second bifurcation')
cbar_alen = subfigs_act.colorbar(axs_act[3*(n_thresh)+1].images[0], cax=axs_act[3*n_thresh+2],shrink=0.5)
cbar_alen.set_label('number of bifurcations')

cbar_m1 = subfigs_act.colorbar(axs_mov[n_thresh-1].images[0], cax=axs_mov[n_thresh])
cbar_m1.set_label('angle of first bifurcation')
cbar_m2 = subfigs_act.colorbar(axs_mov[2*n_thresh].images[0], cax=axs_mov[2*n_thresh+1])
cbar_m2.set_label('angle of second bifurcation')
cbar_mlen = subfigs_act.colorbar(axs_mov[3*n_thresh+1].images[0], cax=axs_mov[3*n_thresh+2])
cbar_mlen.set_label('number of bifurcations')

subfigs_act.suptitle("Activity threshold analysis for angle 1, angle 2, and number of bifurcations")
subfigs_mov.suptitle("Distance threshold analysis for angle 1, angle 2, and number of bifurcations")

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

#print(target_df)
#print(phase_df)
#print(time_df)
#print(angle_df)
#print(angle_2_df)
#print(angle_len_df)

subfigs_metric.tight_layout()
subfigs_act.tight_layout()
subfigs_mov.tight_layout()
plt.show()
