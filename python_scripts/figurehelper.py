import matplotlib.pyplot as plt
import numpy as np

N = 100
angles = np.linspace(0,2*np.pi,N+1)
angles = angles[:-1]
x_base = np.cos(angles)
y_base = np.sin(angles)

targ_x = [25,75]
targ_y = [25,25]

ampl = 0.3
s1 = 0.05
s2 = 0.6
def get_activity(ampl,sigma,targ_angs,N,distAB):
    print(f"targ angs: {targ_angs}, distab: {distAB}, ampl: {ampl}")
    activity = np.zeros(N)
    activity = activity + [1]*N
    if distAB > 0:
        ampl = ampl*(np.exp(-1*distAB/100))
    print(f"adjusted ampl: {ampl}")
    for a in range(N):
        for tang in targ_angs:
            dAng = abs(angles[a]-tang)
            if dAng > np.pi:
                dAng = 2*np.pi - dAng  
            contr = ampl*np.exp(-0.5*(dAng**2)/sigma**2)
            activity[a] += contr

    x_act = activity * np.cos(angles)
    y_act = activity * np.sin(angles)
    #print(x_act,y_act)
    return x_act,y_act

x1 = 50
y1 = 0
x2 = 50
y2 = 15
d1 = np.sqrt(25**2+25**2)
d2 = np.sqrt(10**2+25**2)
t1_ang_1 = np.atan2(targ_y[0]-y1,targ_x[0]-x1)
t2_ang_1 = np.atan2(targ_y[1]-y1,targ_x[1]-x1)
t1_ang_2 = np.atan2(targ_y[0]-y2,targ_x[0]-x2)
t2_ang_2 = np.atan2(targ_y[1]-y2,targ_x[1]-x2)

x_act_1,y_act_1 = get_activity(ampl,s1,[t1_ang_1,t2_ang_1],N,0)
x_act_2, y_act_2 = get_activity(ampl,s1,[t1_ang_2,t2_ang_2],N,0)

x_act_dd1,y_act_dd1 = get_activity(ampl,s1,[t1_ang_1,t2_ang_1],N,d1)
x_act_dd2, y_act_dd2 = get_activity(ampl,s1,[t1_ang_2,t2_ang_2],N,d2)

x_act_1s2,y_act_1s2 = get_activity(ampl,s2,[t1_ang_1,t2_ang_1],N,0)
x_act_2s2, y_act_2s2 = get_activity(ampl,s2,[t1_ang_2,t2_ang_2],N,0)

x_act_dd1s2,y_act_dd1s2 = get_activity(ampl,s2,[t1_ang_1,t2_ang_1],N,d1)
x_act_dd2s2, y_act_dd2s2 = get_activity(ampl,s2,[t1_ang_2,t2_ang_2],N,d2)


# one plot, 2 subplots
fig = plt.figure(layout='constrained',figsize=(13,6))
subfigs = fig.subfigures(1,2, wspace=0.1)
axs0 = subfigs[0].subplots(1,1)
axs0.scatter(targ_x,targ_y,c='red')
axs0.scatter(x1,y1,c='blue')
axs0.scatter(x2,y2,c='green')
axs1 = subfigs[1].subplots(2,2)
axs1 = axs1.flatten()
axs1[0].scatter(x_base,y_base,s=2)
axs1[0].plot(x_act_1,y_act_1,c='blue')
axs1[0].plot(x_act_2,y_act_2,c='green')
axs1[0].set_xticks([])
axs1[0].set_yticks([])
axs1[1].scatter(x_base,y_base,s=2)
axs1[1].plot(x_act_dd1,y_act_dd1,c='blue')
axs1[1].plot(x_act_dd2,y_act_dd2,c='green')
axs1[1].set_xticks([])
axs1[1].set_yticks([])
axs1[2].scatter(x_base,y_base,s=2)
axs1[2].plot(x_act_1s2,y_act_1s2,c='blue')
axs1[2].plot(x_act_2s2,y_act_2s2,c='green')
axs1[2].set_xticks([])
axs1[2].set_yticks([])
axs1[3].scatter(x_base,y_base,s=2)
axs1[3].plot(x_act_dd1s2,y_act_dd1s2,c='blue')
axs1[3].plot(x_act_dd2s2,y_act_dd2s2,c='green')
axs1[3].set_xticks([])
axs1[3].set_yticks([])
plt.show()