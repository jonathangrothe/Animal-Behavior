import numpy as np
import matplotlib.pyplot as plt
import math
import time
import matplotlib.animation as animation

# ---------- Simulation code!! ----------
def simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag,periodic_flag,rEgo,rEgoTarget,Egonumber,
                            distf,adistf,J,beta,h0,h_b,dt,v0,v0t,sigma,hColl,rColl,
                            initialx,initialy,initialxt,initialyt,plot,stop,stopping_dist=0.1):
    '''
    The code to run a single simulation of the ring attractor model. 
    Designed to work with any number of agents but so far I've only really focused on one agent, which impacts stopping distance and v0 right now. 
    Will need to change the logic of the stopping distance code and expand v0 to a list when changing this code.

    Parameters:
        N: an integer that is the number of neurons in the ring attractor
        L: a float or integer that is the width and length of the arena
        T: an integer that is the maximum number of time steps
        ntargets: an integer that is the number of targets
        allocentricFlag: a boolean that represents if we are using an allocentric coordinate system or not
        periodic_flag: a boolean that represents if the arena is periodic (if we go past the end of the arena should we come out the other side)
        rEgo: a float that represents how close an agent needs to be to another agent to consider switching from an allocentric representation to an egocentric representation
        rEgoTarget: a float that represents how close an agent needs to be to a target to consider switching from an allocentric representation to an egocentric representation
        Egonumber: an integer which represents the number of times within rEgo or rEgotarget required to induce the switch from allocentric to egocentric
        distf: a boolean which controls whether or not the distance effects the strength of the input signal
        adistf: a float which controls the exponential input signal decay if distf is 1
        J: a 100x100 array that contains the connectivity profile between neurons. Each value is between -1 and 1, with 1 being more connected.
        Each row is that neurons connectivity profile, and each entry is the connectivity relationship to the neuron corresponding to the column.
        beta: a float or integer which represents the neural noise
        h0: a list of floats which represent the attractiveness for each target, and then (if applicaple) the attractiveness of each agent
        h_b: a float which represents the base inhibition, which is subtracted from the activity of all neurons
        dt: a float which represents the activity step size for each time step
        v0: a float which represents the velocity of the agent
        v0t: a list of floats which represent the velocities of the targets
        sigma: a float which represents the width of the input signal
        hColl: a float which is the updated input signal when the agent is within rColl of another agent, designed to avoid collisions within other agents
        rColl: a float which the distance within which collision avoidance behavior is activated
        initialx: a list of floats which contain the starting x positions of the agents
        initialy: a list of floats which contain the starting y positions of the agents
        intialxt: a list of floats which contain the starting x positions of the targets
        initialyt: a list of floats which contain the starting y positions of the targets
        plot: a boolean which controls whether or not to plot the trajectory every 100 tsteps
        stop: a boolean which controls whether or not to stop the simulation when the agent reaches a target (note: right now should be False if there is more than one agent)
        stopping_dist: a float which controls how close to a target the agent has to get to stop the simulation when stop is True

    Returns: 
        headings: a numpy array of size (nagents x tsteps) which contains each agent's heading in polar coordinates at each point in the simulation 
                  (tsteps is the total number of steps the simulation runs for)
        xPos: a numpy array of size (nagents x tsteps) which contains each agent's x position at each point in the simulation
        yPos: a numpy array of size (nagents x tsteps) which contains each agent's y position at each point in the simulation
        targetXPos: if any target has a v0t above 0, a numpy array of size (ntargets x tsteps) which contains each target's x position at each point in the simulation
                    if all targets have v0t = 0, a numpy array of size (ntargets x 1) which contains each targets initial (and constant) x position
        targetYPos: if any target has a v0t above 0, a numpy array of size (ntargets x tsteps) which contains each target's y position at each point in the simulation
                    if all targets have v0t = 0, a numpy array of size (ntargets x 1) which contains each targets initial (and constant) y position
        uArray: a numpy array of size (N, nagents, tsteps) which contains the state of each neuron for each agent at each time step in the simulation
    '''

    # -------- INITIALIZATIONS --------

    def fAct(u,beta):
        # a function which introduces neural noise
        return ((1+np.tanh(beta * u))/2)

    alpharing0 = np.linspace(0,2*np.pi, N+1)
    alpharing0 = alpharing0[:-1]
    alpharing = np.zeros((nagents,N))
    for a in range (nagents):
        alpharing[a,:] = alpharing0
    
    figure = plt.figure(layout='constrained',figsize=(12,8),num=3)
    subfigs = figure.subfigures(2,3, wspace=0.1)
    color_counter = 0
    plasma_r = plt.colormaps['plasma_r']
    steps = [30,100,250,200,300,350,400,500,600]
    colors = plasma_r.resampled(9)(np.linspace(0,1,9))
    uaold_plot = subfigs[0,0].subplots(1,1)
    subfigs[0,0].suptitle("uaOld")
    faold_plot = subfigs[0,1].subplots(1,1)
    subfigs[0,1].suptitle("faOld")
    netring_plot = subfigs[0,2].subplots(1,1)
    subfigs[0,2].suptitle("net ring")
    extern_plot = subfigs[1,0].subplots(1,1)
    subfigs[1,0].suptitle("external input")
    du_plot = subfigs[1,1].subplots(1,1)
    subfigs[1,1].suptitle("du")
    uanew_plot = subfigs[1,2].subplots(1,1)
    subfigs[1,2].suptitle("uanew")

    details_plot = False
    uArray = np.zeros((N, nagents, T+1))
    u0 = 0.2*np.random.randn(N,nagents) 
    uArray[:,:,0] = u0

    xPos = np.zeros((nagents,T+1))
    yPos = np.zeros((nagents,T+1))
    headings = np.zeros((nagents,T+1))
    for a in range(nagents):
        xPos[a,0] = initialx[a]
        yPos[a,0] = initialy[a]
        headings[a,0] = 2*math.pi*np.random.rand() 
        if allocentricFlag == 0:
            alpharing[a,:] = np.mod(alpharing[a,:]+headings[a,0], 2*np.pi)
    if v0t.any():
        targetXPos = np.zeros((ntargets, T+1))
        targetYPos = np.zeros((ntargets, T+1))
        for targ in range(ntargets):
            targetXPos[targ,0] = initialxt[targ]
            targetYPos[targ,0] = initialyt[targ]
    else:
        targetXPos = initialxt
        targetYPos = initialyt
    true_neurons = np.zeros((ntargets,T))
    adj_ampl = np.zeros((ntargets,T))
    if plot == True:
        plt.figure(1)
        plt.gca().set_aspect('equal', adjustable='box')
    activated = False
    times = []
    fa_data = np.zeros((N,T))
    netring_data = np.zeros((N,T))
    extern_data = np.zeros((N,T))
    du_data = np.zeros((N,T))
    for tstep in range(0,T):
        step_time_start = time.perf_counter()
        # -------- STEP A --------

        Iextern = np.zeros((N,nagents))
        Egocentric = np.zeros(nagents)
        
        # input of other agents
        for a in range(nagents):
            xa = xPos[a,tstep]
            ya = yPos[a,tstep]
            for b in range(nagents):
                if b == a:
                    continue
                xb = xPos[b,tstep]
                yb = yPos[b,tstep]
                dx = xb - xa
                dy = yb - ya
                if periodic_flag == 1:
                    if abs(dx) > L / 2:
                        dx -= math.copysign(L, dx)
                    if abs(dy) > L / 2:
                        dy -= math.copysign(L, dy)
                distAB = np.sqrt(dx**2 + dy**2)
                ampl = h0[ntargets+b]
                if distf != 0: 
                    ampl = ampl*(np.exp(-adistf*distAB/L))
                if distAB < rColl:
                    ampl = hColl
                if distAB < rEgo:
                    Egocentric[a] = Egocentric[a]+1
                    print("entered switch")
                angleAB = np.atan2(dy,dx)
                if angleAB < 0:
                    angleAB = angleAB + 2*np.pi
                for i in range(N):
                    dAng = np.abs(alpharing[a,i] - angleAB)
                    if dAng > np.pi:
                        dAng = 2*np.pi - dAng
                    Iextern[i,a] = Iextern[i,a] + ampl*np.exp(-0.5*(dAng**2)/(sigma**2))
        
        # input of targets
        for a in range(nagents):
            xa = xPos[a,tstep]
            ya = yPos[a,tstep]
            for ttarg in range(ntargets):
                if v0t.any():
                    xt = targetXPos[ttarg,tstep]
                    yt = targetYPos[ttarg,tstep]
                else: 
                    xt = targetXPos[ttarg]
                    yt = targetYPos[ttarg]
                dx = xt - xa
                dy = yt - ya
                if periodic_flag == 1:
                    if abs(dx) > L / 2:
                        dx -= math.copysign(L, dx)
                    if abs(dy) > L / 2:
                        dy -= math.copysign(L, dy)
                distAB = np.sqrt(dx**2 + dy**2)
                if distAB < rEgoTarget:
                    Egocentric[a] = Egocentric[a] + 1
                    print("entered switch")
                ampl = h0[ttarg]
                if distf != 0: 
                    ampl = ampl*(np.exp(-adistf*distAB/L))
                adj_ampl[ttarg,tstep] = ampl
                angleAB = np.atan2(dy,dx)
                if angleAB < 0:
                    angleAB = angleAB + 2*np.pi
                for i in range(N):
                    dAng = abs(alpharing[a,i]-angleAB)
                    if dAng > np.pi:
                        dAng = 2*np.pi - dAng
            
        # -------- STEP B --------
        for a in range(nagents):
            uaOld = uArray[:,a,tstep]
            faOld = fAct(uaOld,beta)
            netRing = (1/N) * (J @ faOld) 
            dU = -uaOld + netRing - h_b + Iextern[:,a]
            uaNew = uaOld + dt * dU
            uArray[:,a,tstep+1] = uaNew
            fa_data[:,tstep] = faOld
            netring_data[:,tstep] = netRing
            du_data[:,tstep] = dU
            extern_data[:,tstep] = Iextern[:,a]
            if tstep in steps:
                details_plot = True
                color = colors[color_counter]
                color_counter += 1
            if details_plot:
                print(f"time: {tstep}")
                uaold_plot.plot(uaOld,c=color)
                faold_plot.plot(faOld,c=color)
                netring_plot.plot(netRing,c=color)
                extern_plot.plot(Iextern[:,a],c=color)
                du_plot.plot(dU,c=color)
                uanew_plot.plot(uaNew,c=color)
                details_plot = False
                print(f"xpos: {xPos[:,tstep]}, ypos: {yPos[:,tstep]}")
            
        # -------- STEP C --------
        for a in range(nagents):
            uaNow = uArray[:,a,tstep+1]
            faNow = fAct(uaNow,beta)
            faPos = faNow.copy()
            faPos[faPos < 0] = 0
            cx = 0
            cy = 0
            for i in range(N):
                cx = cx + (faPos[i] * np.cos(alpharing[a,i]))
                cy = cy + (faPos[i] * np.sin(alpharing[a,i]))
            newAngle = 0
            if (np.abs(cx) < 1e-9) or (np.abs(cy) < 1e-9):
                newAngle = headings[a,tstep]
            else:
                newAngle = np.atan2(cy,cx)
                if newAngle < 0:
                    newAngle = newAngle + 2*np.pi
            headings[a,tstep+1] = newAngle
            if (tstep > 100) and (0 < newAngle < np.pi/2 or activated):
                delta_x = xPos[a,tstep]-xPos[a,tstep-1]
                delta_y = yPos[a,tstep]-yPos[a,tstep-1]
                #print(f'tstep: {tstep}, heading: {newAngle}, distance: {targ_differences}, target neurons: {true_neurons[0,tstep]:.4f}, {true_neurons[1,tstep]:.4f}, {true_neurons[2,tstep]:.4f}')
                if not activated: 
                    activated = True
            if allocentricFlag == 0:
                alpharing[a,:] = np.mod(alpharing[a,:] - headings[a,tstep] + headings[a,tstep+1],2*np.pi)
            elif Egocentric[a] >= Egonumber:
                alpharing[a,:] = np.mod(alpharing[a,:]-headings[a,tstep] + headings[a,tstep+1], 2*np.pi)
        
        # -------- STEP D --------
        for a in range(nagents):
            oldx = xPos[a,tstep]
            oldy = yPos[a,tstep]
            cx_d = 0
            cy_d = 0
            for i in range(N):
                val = fAct(uArray[i,a,tstep+1],beta) 
                if val < 0:
                    val = 0
                cx_d = cx_d + val * np.cos(alpharing[a,i])
                cy_d = cy_d + val * np.sin(alpharing[a,i])
            newx = oldx + dt * v0 * cx_d
            newy = oldy + dt * v0 * cy_d
            if periodic_flag == 1:
                newx = np.mod(newx,L)
                newy = np.mod(newy,L)
            if newx < 0:
                newx = 0
            elif newx > L:
                newx = L
            if newy < 0:
                newy = 0
            elif newy > L:
                newy = L
            xPos[a,tstep+1] = newx
            yPos[a,tstep+1] = newy

        # -------- STEP E --------
        if v0t.any():
            for ttarg in range(ntargets):
                oldXT = targetXPos[ttarg,tstep]
                oldYT = targetYPos[ttarg,tstep]
                newXT = oldXT + v0t[ttarg] * np.sign(np.random.randn())
                newYT = oldYT + v0t[ttarg] * np.sign(np.random.randn())
                if periodic_flag == 1:
                    newx = np.mod(newXT,L)
                    newy = np.mod(newYT,L)
                if newXT < 0:
                    newXT = 0
                elif newXT > L:
                    newXT = L
                if newYT < 0:
                    newYT = 0
                elif newYT > L:
                    newYT = L
                targetXPos[ttarg, tstep+1] = newXT
                targetYPos[ttarg, tstep+1] = newYT

        # -------- STEP F --------
        if plot == True:
            plot_col= 'blue'
            if allocentricFlag == 1:
                plot_col = 'green'
            if tstep % 100 == 0:
                plt.scatter(
                    xPos[:, tstep + 1],
                    yPos[:, tstep + 1],
                    s = 10,
                    color = plot_col
                )
                xtarg = 0
                ytarg = 0
                if v0t.any():
                    xtarg = targetXPos[:, tstep + 1]
                    ytarg = targetYPos[:, tstep + 1]
                else:
                    xtarg = targetXPos
                    ytarg = targetYPos
                if ntargets > 0:
                    plt.scatter(
                        xtarg,
                        ytarg,
                        s = 10,
                        marker = 's',
                        c = [[0.8, 0, 0.2]], 
                        )
                plt.xlim(0, L)
                plt.ylim(0, L)
                plt.title(f"Time step {tstep}, allocentric:{allocentricFlag}, sigma: {sigma}")
                plt.pause(0.001)  

        # -------- STEP G --------
        if stop == True:
            targ_x = 0
            targ_y = 0
            if v0t.any():
                targ_x = targetXPos[:,tstep]
                targ_y = targetYPos[:,tstep]
            else:
                targ_x = targetXPos
                targ_y = targetYPos
            x_dist = np.abs(xPos[0,tstep] - targ_x)
            y_dist = np.abs(yPos[0,tstep] - targ_y)
            total_dist = (x_dist**2)+(y_dist**2)
            if np.min(total_dist) <= stopping_dist:
                headings = headings[:,:tstep]
                xPos = xPos[:,:tstep]
                yPos = yPos[:,:tstep]
                if v0t.any():
                    targetXPos = targetXPos[:,:tstep]
                    targetYPos = targetYPos[:,:tstep]
                uArray = uArray[:,:,:tstep]
                #plt.show()
                '''
                data = {'uaold': uArray[:,0,:-1],
                        'faold':fa_data[:,:tstep-1],
                        'netring':netring_data[:,:tstep-1],
                        'extern':extern_data[:,:tstep-1],
                        'du':du_data[:,:tstep-1],
                        'uanew':uArray[:,0,1:]}
                print(f"tstep: {tstep},len:{len(data['netring'][0,:])}")
                x = np.arange(100)
                t = np.arange(tstep-1) 
                fig, axes = plt.subplots(3,2,layout='constrained',figsize=(12,8),num=4)
                axes = axes.flatten()
                lines = {}
                for ax, (name, arr) in zip(axes, data.items()):
                    (line,) = ax.plot([], [])
                    lines[name] = line
                    ax.set_xlim(x.min(), x.max())
                    ax.set_ylim(arr.min(), arr.max())
                    ax.set_title(name)
                fig.suptitle("t = 0")
                def update(frame):
                    for name, line in lines.items():
                        line.set_data(x,data[name][:,frame])
                    fig.suptitle(f"t={t[frame]}")
                    return list(lines.values())
                
                
                ani = animation.FuncAnimation(fig, update, frames=tstep-1, interval=100, blit=False)
                ani.save("example_animation.gif", writer="pillow", fps=30, dpi=150)
                '''
                return headings, xPos, yPos, targetXPos, targetYPos, uArray
            
        if plot == True:
            plt.show(block = False)
        step_end_time = time.perf_counter()
        times.append(step_end_time-step_time_start)
    '''
    data = {'uaold': uArray[:,0,:-1],
            'faold':fa_data[:,:tstep-1],
            'netring':netring_data[:,:tstep-1],
            'extern':extern_data[:,:tstep-1],
            'du':du_data[:,:tstep-1],
            'uanew':uArray[:,0,1:]}
    x = np.arange(100)
    t = np.arange(tstep-1) 
    fig, axes = plt.subplots(3,2,layout='constrained',figsize=(12,8),num=4)
    axes = axes.flatten()
    lines = {}
    for ax, (name, arr) in zip(axes, data.items()):
        (line,) = ax.plot([], [])
        lines[name] = line
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(arr.min(), arr.max())
        ax.set_title(name)
    fig.suptitle("t = 0")
    def update(frame):
        for name, line in lines.items():
            line.set_data(x,data[name][:,frame])
        fig.suptitle(f"t={t[frame]}")
        return list(lines.values())
    
    
    ani = animation.FuncAnimation(fig, update, frames=tstep-1, interval=100, blit=False)
    ani.save("example_animation.gif", writer="pillow", fps=30, dpi=150)
    '''
    return headings, xPos, yPos, targetXPos, targetYPos, uArray

