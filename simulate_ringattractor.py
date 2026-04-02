import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- Simulation code!! ----------
def simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag,periodic_flag,rEgo,rEgoTarget,Egonumber,
                            distf,adistf,J,beta,h0,h_b,dt,v0,v0t,sigma,hColl,rColl,
                            initialx,initialy,initialxt,initialyt,plot=True,stop=False,stopping_dist=0.1):

    # -------- INITIALIZATIONS WITHIN THE SIMULATION --------

    alpharing0 = np.linspace(0,2*np.pi, N+1)
    alpharing0 = alpharing0[:-1]
    alpharing = np.zeros((nagents,N))
    for a in range (nagents):
        alpharing[a,:] = alpharing0

    uArray = np.zeros((N, nagents, T+1))
    u0 = 0.05*np.random.randn(N,nagents) #initialize randomly
    uArray[:,:,0] = u0

    xPos = np.zeros((nagents,T+1))
    yPos = np.zeros((nagents,T+1))
    headings = np.zeros((nagents,T+1))
    for a in range(nagents):
        xPos[a,0] = initialx[a]
        yPos[a,0] = initialy[a]
        headings[a,0] = 0
        alpharing[a,:] = np.mod(alpharing[a,:]+headings[a,0], 2*np.pi)

    targetXPos = np.zeros((ntargets, T+1))
    targetYPos = np.zeros((ntargets, T+1))
    for targ in range(ntargets):
        targetXPos[targ,0] = initialxt[targ]
        targetYPos[targ,0] = initialyt[targ]

    plt.figure(1)
    plt.gca().set_aspect('equal', adjustable='box')

    for tstep in range(0,T):

        # -------- STEP A --------

        Iextern = np.zeros((N,nagents))
        Egocentric = np.zeros(nagents)
        
        #contribution of other agents
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
        
        for a in range(nagents):
            xa = xPos[a,tstep]
            ya = yPos[a,tstep]
            for ttarg in range(ntargets):
                xt = targetXPos[ttarg,tstep]
                yt = targetYPos[ttarg,tstep]

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
                
                angleAB = np.atan2(dy,dx)
                
                if angleAB < 0:
                    angleAB = angleAB + 2*np.pi
                for i in range(N):
                    dAng = abs(alpharing[a,i]-angleAB)
                    if dAng > np.pi:
                        dAng = 2*np.pi - dAng
                    Iextern[i,a] = Iextern[i,a] + ampl*np.exp(-0.5*(dAng**2)/sigma**2)

        # -------- STEP B --------
        for a in range(nagents):
            uaOld = uArray[:,a,tstep]
            faOld = fAct(uaOld,beta)
            netRing = (1/N) * (J @ faOld) 
            dU = -uaOld + netRing - h_b + Iextern[:,a]
            uaNew = uaOld + dt * dU
            uArray[:,a,tstep+1] = uaNew

        # -------- STEP C --------
        for a in range(nagents):
            uaNow = uArray[:,a,tstep+1]
            faNow = fAct(uaNow,beta)
            faPos = faNow.copy()
            faPos[faPos < 0] = 0

            #calculate centers
            cx = 0
            cy = 0
            for i in range(N):
                cx = cx + (faPos[i] * np.cos(alpharing[a,i]))
                cy = cy + (faPos[i] * np.sin(alpharing[a,i]))
            newAngle = 0
            #if the centers are close to zero use old heading 
            if (np.abs(cx) < 1e-9) and (np.abs(cy) < 1e-9):
                newAngle = headings[a,tstep]
            else:
                newAngle = np.atan2(cy,cx)
                if newAngle < 0:
                    newAngle = newAngle + 2*np.pi
            headings[a,tstep+1] = newAngle

            if allocentricFlag == 0:
                alpharing[a,:] = np.mod(alpharing[a,:] - headings[a,tstep] + newAngle,2*np.pi)
            elif Egocentric[a] >= Egonumber:
                alpharing[a,:] = np.mod(alpharing[a,:]-headings[a,tstep] + newAngle, 2*np.pi)

        # -------- STEP D --------
        for a in range(nagents):
            oldx = xPos[a,tstep]
            oldy = yPos[a,tstep]

            #calculating the centers again
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
            

        # -------- STEP F - VIZ --------
        if plot == True:
            plot_col= 'blue'
            if allocentricFlag == 1:
                plot_col = 'green'

            if tstep % 100 == 0:
                # Plot agents
                plt.scatter(
                    xPos[:, tstep + 1],
                    yPos[:, tstep + 1],
                    s = 10,
                    color = plot_col
                )

                # Plot targets
                if ntargets > 0:
                    plt.scatter(
                        targetXPos[:, tstep + 1],
                        targetYPos[:, tstep + 1],
                        s = 10,
                        marker = 's',
                        c = [[0.8, 0, 0.2]], 
                        )

                plt.xlim(0, L)
                plt.ylim(0, L)
                plt.title(f"Time step {tstep}, allocentric:{allocentricFlag}, h0:{h0}, hb: {h_b}, beta: {beta}, sigma: {sigma}")
                plt.pause(0.001)  

        # -------- STEP G - STOP --------
        if stop == True:
            for targ in range(ntargets):
                targ_x = targetXPos[targ,tstep]
                targ_y = targetYPos[targ,tstep]
                x_dist = np.abs(xPos[0,tstep] - targ_x)
                y_dist = np.abs(yPos[0,tstep] - targ_y)
                total_dist = np.sqrt((x_dist**2)+(y_dist**2))
                if total_dist <= stopping_dist:
                    print(f"target reached at: {tstep}")
                    headings = headings[:,:tstep+1]
                    xPos = xPos[:,:tstep+1]
                    yPos = yPos[:,:tstep+1]
                    targetXPos = targetXPos[:,:tstep+1]
                    targetYPos = targetYPos[:,:tstep+1]
                    return headings, xPos, yPos, targetXPos, targetYPos
                
        if plot == True:
            plt.show(block = False)

    return headings, xPos, yPos, targetXPos, targetYPos

#function for creating an evenly spaced grid
def create_grid(ntargets,ncols,L):
    '''
    ntargets: number of targets
    ncols: number of columns in the grid
    L: total length of the space
    returns: 2 lists (one for x one for y) of coordinates arranged in a grid
    '''
    initialxt = np.zeros(ntargets)
    initialyt = np.zeros(ntargets)  
    targets_percol = ntargets//ncols
    for col in range(ncols):
        for ti in range(targets_percol):
            initialxt[targets_percol*col+ti] = col*(L/(ncols+1)) + (L/(ncols+1))
            #this spreads the targets equally vertically,
            #for a grid need to do this several times
            initialyt[targets_percol*col+ti] = ti*(L/(targets_percol+1)) + (L/(targets_percol+1))
    return initialxt, initialyt

#function for creating an evenly spaced circle
def create_circle(ntargets,radius,L):
    '''
    ntargets: number of targets
    radius: radius of the circle
    L: total length of the space
    returns: 2 lists (one for x and one for y) of coordinates arranged in a circle
    '''
    angles = np.linspace(0,2*np.pi,ntargets,endpoint=False)
    x_coords = radius * np.cos(angles) + L/2
    y_coords = radius * np.sin(angles) + L/2
    return x_coords, y_coords

#activation function
def fAct(u,beta):
    '''
    u: Value we are activating
    beta: Noise parameter
    returns: value put through the activation function
    '''
    return ((1+np.tanh(beta * u))/2)




