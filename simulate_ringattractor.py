import numpy as np
import matplotlib.pyplot as plt

# ---------- Simulation code!! ----------
def simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag,rEgo,rEgoTarget,Egonumber,
                            distf,adistf,J,beta,h0,h_b,dt,v0,v0t,sigma,hColl,rColl,
                            initialx,initialy,initialxt,initialyt):

    # -------- INITIALIZATIONS WITHIN THE SIMULATION --------

    #setting up the alpharing
    alpharing0 = np.linspace(0,2*np.pi, N+1)
    alpharing0 = alpharing0[:-1]
    alpharing = np.zeros((nagents,N))
    for a in range (nagents):
        alpharing[a,:] = alpharing0

    #setting up array to hold ring states at different times
    uArray = np.zeros((N, nagents, T+1))
    #u0 = np.zeros((N,nagents))
    u0 = 0.05*np.random.randn(N,nagents) #initialize randomly
    for i in range(len(u0)):
        if (i < len(u0)/8) or (3*len(u0)/8 < i < len(u0)/2 ):
            u0[i] = 0.05
        if len(u0)/8 < i < 3*len(u0)/8:
            u0[i] = 0.1
    #print(f"u0: {u0}")
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
                #Periodic flag add (not yet)
                distAB = np.sqrt(dx**2 + dy**2)
                ampl = h0[ntargets+b]
                if distf != 0: 
                    #don't technically need this, but original code runs one sim with distf =0
                    ampl = ampl*(np.exp(-adistf*distAB/L))
                if distAB < rColl:
                    ampl = hColl
                if distAB < rEgo:
                    Egocentric[a] = Egocentric[a]+1
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

                #ADD PERIODIC FLAG 

                distAB = np.sqrt(dx**2 + dy**2)
                if distAB < rEgoTarget:
                    Egocentric[a] = Egocentric[a] + 1
                
                ampl = h0[ttarg]
                if distf != 0: 
                    #don't technically need this, but original code runs one sim with distf =0
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
            faPos = faNow
            faPos[faPos < 0] = 0

            #calculate centers
            cx = 0
            cy = 0
            for i in range(N):
                cx = cx + (faPos[i] * np.cos(alpharing[a,i]))
                cy = cy + (faPos[i] * np.sin(alpharing[a,i]))
            #print(f"cx - step c: {cx}")
            #print(f"cy - step c: {cy}")
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
                #print(f"Ego update - step C: {alpharing[a,:]}")
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
            #print(f"cx - step D: {cx_d}")
            #print(f"cy - step D: {cy_d}")
            newx = oldx + dt * v0 * cx_d
            newy = oldy + dt * v0 * cy_d

            #add periodicflag later (maybe?)
            if newx < 0:
                newx = 0
            elif newx > L:
                newx = L
            if newy < 0:
                newy = 0
            elif newy > L:
                newy = L

            #print(f"newx: {newx}")
            #print(f"newy: {newy}")
            xPos[a,tstep+1] = newx
            yPos[a,tstep+1] = newy
        
        # -------- STEP E --------

        for ttarg in range(ntargets):
            oldXT = targetXPos[ttarg,tstep]
            oldYT = targetYPos[ttarg,tstep]

            newXT = oldXT + v0t[ttarg] * np.sign(np.random.randn())
            newYT = oldYT + v0t[ttarg] * np.sign(np.random.randn())

            #add periodic flag here

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

        if tstep % 50 == 0:
            # Plot agents
            plt.scatter(
                xPos[:, tstep + 1],
                yPos[:, tstep + 1],
                s = 10,
                c = [[0, 0.2, 0.8]]
            )

            # Plot targets
            if ntargets > 0:
                plt.scatter(
                    targetXPos[:, tstep + 1],
                    targetYPos[:, tstep + 1],
                    s = 10,
                    marker = 's',
                    c = [[0.8, 0, 0.2]]
                    )

            plt.xlim(0, L)
            plt.ylim(0, L)
            plt.title(f"Time step {tstep} (beta={beta:.2f})")
            plt.pause(0.001)  
        
    plt.show()

# --------  PARAMETERS --------

#number of neurons in the alpha ring
N = 100

#total rectangle?
L = 100

#number of time steps
T = 5000

#number of targets
ntargets = 20

#number of agents: 
nagents = 1

#allocentric flag
allocentricFlag = 1

#rego
rEgo = 0

#in the original code the simulation is run 5 times with [0, 0.5, 1, 2, 4] as values
rEgoTarget = 0

#egonumber
Egonumber = 1

#collision avoidance:
hColl = -10

rColl = 0

#this determines the dependence on distance
#in the original code the simulation runs it on 0 once and then 1 five times
distf = 1

#this determines the "characteristic length of signal decay",
#original code iterates through [1, 1, 2, 4, 8, 16]
adistf = 1

#attraction
h0s = [0.3,0.3,0.3]

#hbase
h_b = 0

#dt: step size I believe
dt = 0.1

#v0: velocity I believe
v0 = 0.05

#v0t: velocity of targets I believe
#in the original code it is initialized to zero but that doesn't really make sense to me
v0t = np.zeros(ntargets)
for i in range(ntargets):
    v0t[i] = 0.01

#width of the gauss bump 
#seemed most interesting at 0.2?
sigma = 0.2

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

#the agent's initial x,y and heading
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = L/2
    initialy[a] = L/2

#the target's initial position
initialxt = np.zeros(ntargets)
initialyt = np.zeros(ntargets)
ncols = 4
targets_percol = ntargets//ncols
for col in range(ncols):
    for ti in range(targets_percol):
        initialxt[targets_percol*col+ti] = col*(L/(ncols+1)) + (L/(ncols+1))
        #this spreads the targets equally vertically,
        #for a grid need to do this several times
        initialyt[targets_percol*col+ti] = ti*(L/(targets_percol+1)) + (L/(targets_percol+1))

#beta: noise parameter
beta = 80

#activation function
def fAct(u,beta):
    return ((1+np.tanh(beta * u))/2)

for h in range(len(h0s)):
    h0 = np.zeros(ntargets+nagents)
    for i in range(ntargets):
        h0[i] = h0s[h]
    for a in range(nagents):
        h0[ntargets+a] = h0s[h]
    simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag,rEgo,rEgoTarget,Egonumber,
                            distf,adistf,J,beta,h0,h_b,dt,v0,v0t,sigma,hColl,rColl,
                            initialx,initialy,initialxt,initialyt)





