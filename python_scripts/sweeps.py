import time
import statistics
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
initialxt = [50-15*np.sqrt(3),50,50+15*np.sqrt(3)]
initialyt = [35,80,35]

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
sample_size = 5
sigma_start = 0.14
sigma_finish = 0.26
base_sigma = np.linspace(sigma_start,sigma_finish,num=10)
h0_start = 0.19
h0_finish = 0.24
n_h0 = 10
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
angle_3_grid = []
angle_len_grid = []
angle_2_len_grid = []
angle_3_len_grid = []

for sigma_index in range(len(base_sigma)):
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
    phases = []
    angles = []
    angles_2 = []
    angles_3 = []
    angle_len = []
    angle_2_len = []
    angle_3_len = []
    for i in range(len(target_list)):
        curr_activity = activity_list[i]
        curr_xpos = x_list[i:(i+1)][0]
        curr_ypos = y_list[i:(i+1)][0]
        phase = sim_met.get_bump_type(curr_activity)
        phases.append(phase)
        xt = 'n'
        yt = 'n'
        if target_list[i] >=0:
            xt = initialxt[target_list[i]]
            yt = initialyt[target_list[i]]
        indices, angle = sim_met.get_bifurcation_angle(curr_xpos, curr_ypos, 0.15,25)
        indices_dist9, angles_dist9 = sim_met.get_bifurcation_angle(curr_xpos,curr_ypos,0.25,25)
        indices_dist25, angles_dist25 = sim_met.get_bifurcation_angle(curr_xpos,curr_ypos,0.35,25)
        if len(angle) >0 :
            angles.append(angle[0])
            if angle[0] == np.pi and len(angle) == 1:
                angle_len.append(0)
            else:
                angle_len.append(len(angle))
        else: 
            print("no bif detected")
            angles.append(np.pi)
            angle_len.append(0)
        if len(angles_dist9) > 0:
            angles_2.append(angles_dist9[0])
            if angles_dist9[0] == np.pi and len(angles_dist9) == 1:
                angle_2_len.append(0)
            else:
                angle_2_len.append(len(angles_dist9))
        else:
            angles_2.append(np.pi)
            angle_2_len.append(0)
            print("no bif detected")
        if len(angles_dist25) > 0:
            angles_3.append(angles_dist25[0])
            if angles_dist25[0] == np.pi and len(angles_dist25) ==1:
                angle_3_len.append(0)
            else:
                angle_3_len.append(len(angles_dist25))
        else:
            print("no bif detected")
            angle_3_len.append(0)
            angles_3.append(np.pi)
        

    grid_phases = []
    grid_targets = []
    grid_times = []
    grid_angles = []
    grid_angles_2 = []
    grid_angles_3 = []
    grid_angle_len = []
    grid_angle2_len = []
    grid_angle3_len = []
    for s in range(n_h0):
        sim_phases = phases[s*sample_size:(s+1)*sample_size]
        sim_targets = target_reached[s*sample_size:(s+1)*sample_size]
        sim_times = time_list[s*sample_size:(s+1)*sample_size]
        sim_angles = angles[s*sample_size:(s+1)*sample_size]
        sim_angles2 = angles_2[s*sample_size:(s+1)*sample_size]
        sim_angles3 = angles_3[s*sample_size:(s+1)*sample_size]
        sim_anglelen = angle_len[s*sample_size:(s+1)*sample_size]
        sim_angle2len = angle_2_len[s*sample_size:(s+1)*sample_size]
        sim_angle3len = angle_3_len[s*sample_size:(s+1)*sample_size]
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
        grid_phases.append(np.argmax([n0,n1,n2,n3,nOther]))
        grid_targets.append(reach)
        grid_times.append(np.mean(sim_times))
        grid_angles.append(np.mean(sim_angles))
        grid_angles_2.append(np.mean(sim_angles2))
        grid_angles_3.append(np.mean(sim_angles3))
        grid_angle_len.append(np.mean(sim_anglelen))
        grid_angle2_len.append(np.mean(sim_angle2len))
        grid_angle3_len.append(np.mean(sim_angle3len))

    phase_grid.append(grid_phases)
    target_grid.append(grid_targets)
    time_grid.append(grid_times)
    angle_grid.append(grid_angles)
    angle_2_grid.append(grid_angles_2)
    angle_3_grid.append(grid_angles_3)
    angle_len_grid.append(grid_angle_len)
    angle_2_len_grid.append(grid_angle2_len)
    angle_3_len_grid.append(grid_angle3_len)

    end_analysis_time = time.perf_counter()
    analyzing_time = end_analysis_time - analysis_time
    print(f"Analysis time: {analyzing_time:.6f} seconds")


target_df = pd.DataFrame(target_grid,columns=h0_range,index=base_sigma)
phase_df = pd.DataFrame(phase_grid,columns=h0_range,index=base_sigma)
time_df = pd.DataFrame(time_grid,columns=h0_range,index=base_sigma)
angle_df = pd.DataFrame(angle_grid,columns=h0_range,index=base_sigma)
angle_2_df = pd.DataFrame(angle_2_grid,columns=h0_range,index=base_sigma)
angle_3_df = pd.DataFrame(angle_3_grid,columns=h0_range,index=base_sigma)
angle_len_df = pd.DataFrame(angle_len_grid,columns=h0_range,index=base_sigma)
angle2_len_df = pd.DataFrame(angle_2_len_grid,columns=h0_range,index=base_sigma)
angle3_len_df = pd.DataFrame(angle_3_len_grid,columns=h0_range,index=base_sigma)


subfigs, axs = plt.subplots(nrows=3,ncols=3, figsize = (16,14))
axs = axs.flatten()

bwr_r = plt.colormaps['bwr_r']
discrete_bwrr = bwr_r.resampled(3)(np.linspace(0,1,3))
viridis = plt.colormaps['viridis']
discrete_viridis = viridis.resampled(5)(np.linspace(0,1,5))
cmap1 = mcolors.ListedColormap(discrete_bwrr)
cmap2 = mcolors.ListedColormap(discrete_viridis)
cmap3 = 'inferno_r'
cmap4 = 'plasma'
boundaries_tar = np.arange(-1,3) - 0.5
norm_tar = mcolors.BoundaryNorm(boundaries_tar,cmap1.N)
boundaries_phase = np.arange(5) - 0.5
norm_phase = mcolors.BoundaryNorm(boundaries_phase,cmap2.N)
categories_tar = ['fails', 'outer', 'center']
categories_phase = ['0 bumps', '1 bump', '2 bumps','3 bumps']

im1 = axs[0].imshow(target_df, cmap=cmap1,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im2 = axs[1].imshow(phase_df,cmap=cmap2,origin='lower',extent=[h0_range[0], h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im3 = axs[2].imshow(time_df,cmap=cmap3,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im4 = axs[3].imshow(angle_df,cmap=cmap4,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im5 = axs[4].imshow(angle_2_df,cmap=cmap4,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im6 = axs[5].imshow(angle_3_df,cmap=cmap4,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im7 = axs[6].imshow(angle_len_df,cmap=cmap3,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im8 = axs[7].imshow(angle2_len_df,cmap=cmap3,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
im9 = axs[8].imshow(angle3_len_df,cmap=cmap3,origin='lower',extent=[h0_range[0],h0_range[n_h0-1], base_sigma[0], base_sigma[len(base_sigma)-1]],aspect='auto')
cbar1 = plt.colorbar(im1, ticks=np.arange(-1,2))
cbar1.ax.set_yticklabels(categories_tar)
cbar2 = plt.colorbar(im2, ticks=np.arange(4))
cbar2.ax.set_yticklabels(categories_phase)
cbar3 = plt.colorbar(im3)
cbar3.set_label('Time to target')
cbar4 = plt.colorbar(im4)
cbar4.set_label('angle of first bifurcation (thresh 0.15)')
cbar5 = plt.colorbar(im5)
cbar5.set_label('angle of first bifurcation (thresh 0.25)')
cbar6 = plt.colorbar(im6)
cbar6.set_label('angle of first bifurcation (thresh: 0.25)')
cbar7 = plt.colorbar(im7)
cbar7.set_label('number of bifurcations')
cbar8 = plt.colorbar(im8)
cbar8.set_label('number of bifurcations')
cbar9 = plt.colorbar(im9)
cbar9.set_label('number of bifurcations')

subfigs.suptitle(f"Heatmaps for 2π/3 between targets, average of {sample_size} samples (dist: 25)")
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

print(target_df)
print(phase_df)
print(time_df)
print(angle_df)
print(angle_2_df)
print(angle_len_df)
print(angle2_len_df)
print(angle3_len_df)

plt.tight_layout()
plt.show()
