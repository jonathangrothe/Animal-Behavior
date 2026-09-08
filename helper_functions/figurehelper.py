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


plt.figure(1)
setup_x = [50,50]
setup_y = [20,50]
x_120 = [50-15*np.sqrt(3),50+15*np.sqrt(3)]
y_120 = [5,5]
x_90 = [20,80]
y_90 = [20,20]
x_60 = [50-15*np.sqrt(3),50+15*np.sqrt(3)]
y_60 = [35,35]
x_30 = [35,65]
y_30 = [20+15*np.sqrt(3),20+15*np.sqrt(3)]

plt.scatter(setup_x,setup_y,c='black')
plt.scatter(x_120,y_120,c='red')
plt.scatter(x_90,y_90,c='green')
plt.scatter(x_60,y_60,c='blue')
plt.scatter(x_30,y_30,c='purple')




plt.show()