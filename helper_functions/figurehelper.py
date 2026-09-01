import matplotlib.pyplot as plt
import numpy as np

N = 100
angles = np.linspace(0,2*np.pi,N+1)
angles = angles[:-1]
x_base = np.cos(angles)
y_base = np.sin(angles)

targ_x = [25,75]
targ_y = [25,25]

theta = np.linspace(0,2*np.pi,N+1)
theta = theta[:-1]
J = np.zeros((N,N))
for i in range(N):
    deltah = np.abs(theta - theta[i])
    deltah = np.pi-np.abs(np.pi-deltah)
    J[i,:] = np.cos(np.pi*(deltah/np.pi)**0.5)
    J[i,i] = 0.0
    J = np.squeeze(J)

ampl_large = [0.25,0.25]
ampl_small = [0.2,0.2]
ampl_ext = [0.3,0.3]
s=0.2
beta = 100

alpharing = np.linspace(0,2*np.pi, N+1)
alpharing = alpharing[:-1]

def fAct(u,beta):
    return ((1+np.tanh(beta * u))/2)

def get_activity(ampl,sigma,targ_angs,N,distAB):
    print(f"targ angs: {targ_angs}, distab: {distAB}, ampl: {ampl}")
    activity = np.zeros(N)
    activity = activity
    if distAB > 0:
        ampl = ampl*(np.exp(-1*distAB/100))
    print(f"adjusted ampl: {ampl}")
    for a in range(N):
        for t in range(2):
            tang = targ_angs[t]
            ampl_t = ampl[t]
            dAng = abs(angles[a]-tang)
            if dAng > np.pi:
                dAng = 2*np.pi - dAng  
            contr = ampl_t*np.exp(-0.5*(dAng**2)/sigma**2)
            activity[a] += contr
    return activity

def one_step(act_old,act_ext):
    uaOld = act_old
    faOld = fAct(uaOld,beta)
    netRing = (1/N) * (J @ faOld) 
    dU = -uaOld + netRing - 0.2 + act_ext
    uaNew = uaOld + dU
    return uaNew

def get_heading(act):
    f0 = fAct(act,beta)
    f0_copy = f0.copy()
    f0_copy[f0_copy < 0] = 0
    cx0 = 0
    cy0 = 0
    for i in range(N):
        cx0 = cx0 + (f0_copy[i] * np.cos(alpharing[i]))
        cy0 = cy0 + (f0_copy[i] * np.sin(alpharing[i]))
    heading = np.atan2(cy0,cx0)
    if heading < 0:
        heading = heading + 2*np.pi
    print(f"cx: {cx0}, cy: {cy0}, heading: {heading}")
    return heading

x1 = 50
y1 = 15
#x2 = 50
#y2 = 20
d1 = np.sqrt(25**2+25**2)
d2 = np.sqrt(10**2+25**2)
t1_ang_1 = np.atan2(targ_y[0]-y1,targ_x[0]-x1)
t2_ang_1 = np.atan2(targ_y[1]-y1,targ_x[1]-x1)
#t1_ang_2 = np.atan2(targ_y[0]-y2,targ_x[0]-x2)
#t2_ang_2 = np.atan2(targ_y[1]-y2,targ_x[1]-x2)

extern = get_activity(ampl_ext,s,[t1_ang_1,t2_ang_1],N,0)
init_large = get_activity(ampl_large,s,[t1_ang_1,t2_ang_1],N,0) - 0.2
next_step_large = one_step(init_large,extern)
init_small = get_activity(ampl_small,s,[t1_ang_1,t2_ang_1],N,0) - 0.2
next_step_small = one_step(init_small,extern)

init_large_heading = get_heading(init_large)
il_x2 = x1 + 5*np.cos(init_large_heading)
il_y2 = y1 + 5*np.sin(init_large_heading)

post_large_heading = get_heading(next_step_large)
pl_x2 = x1 + 5*np.cos(post_large_heading)
pl_y2 = y1 + 5*np.sin(post_large_heading)

init_small_heading = get_heading(init_small)
is_x2 = x1 + 5*np.cos(init_small_heading)
is_y2 = y1 + 5*np.sin(init_small_heading)

post_small_heading = get_heading(next_step_small)
ps_x2 = x1 + 5*np.cos(post_small_heading)
ps_y2 = y1 + 5*np.sin(post_small_heading)


# one plot, 2 subplots
fig = plt.figure(layout='constrained',figsize=(10,5))
subfigs = fig.subfigures(2,3, wspace=0.1)
sf0 = subfigs[0][0]
ax0 = sf0.subplots(2,1)
ax0[0].plot(init_large)
ax0[0].set_title("initial activity, large")
ax0[1].plot(next_step_large)
#ax0[2].plot(activity1-activity2)
sf1 = subfigs[0][1]
ax1 = sf1.subplots(3,1)
ax1[0].plot(fAct(init_large,beta))
ax1[1].plot(fAct(next_step_large,beta))
ax1[2].plot(fAct(next_step_large,beta)-fAct(init_large,beta))

sf2 = subfigs[1][0]
ax2 = sf2.subplots(2,1)
ax2[0].plot(init_small)
ax2[1].plot(next_step_small)
#ax2[2].plot(activity_lower1-activity_lower2)
sf3 = subfigs[1][1]
ax3 = sf3.subplots(3,1)
ax3[0].plot(fAct(init_small,beta))
ax3[1].plot(fAct(next_step_small,beta))
ax3[2].plot(fAct(next_step_small,beta)-fAct(init_small,beta))

sf4 = subfigs[0][2]
ax4 = sf4.subplots(1,1)
ax4.scatter(targ_x,targ_y,c='red')
ax4.arrow(x1,y1,il_x2-x1,il_y2-y1,width=0.005,head_width=0.4,head_length=0.2,color='blue')
ax4.arrow(x1,y1,pl_x2-x1,pl_y2-y1,width=0.005,head_width=0.4,head_length=0.2,color='green')

sf5 = subfigs[1][2]
ax5 = sf5.subplots(1,1)
ax5.scatter(targ_x,targ_y,c='red')
ax5.arrow(x1,y1,is_x2-x1,is_y2-y1,width=0.005,head_width=0.4,head_length=0.2,color='blue')
ax5.arrow(x1,y1,ps_x2-x1,ps_y2-y1,width=0.005,head_width=0.4,head_length=0.2,color='green')


plt.show()