import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simulate_ringattractor
import simulation_metrics

# --------  PARAMETERS --------

# --- Setting up the grid ---

#number of neurons in the alpha ring
N = 100

#total size of the grid
L = 100

# --- Geometry-based parameters to change ---

#number of targets
ntargets = 2

#number of agents: 
nagents = 1

#setting up the agent
initialx = np.zeros(nagents)
initialy = np.zeros(nagents)
for a in range(nagents):
    initialx[a] = 20
    initialy[a] = 50
    
#setting up the targets
initialxt = [80,80]
initialyt = [20,80]





# --- Setting up the simulation ---

#number of time steps
T = 5000

#periodic flag
periodicflag = 0

#rego
rEgo = 0

#in the original code the simulation is run 5 times with [0, 0.5, 1, 2, 4] as values
rEgoTarget = 0

#egonumber
Egonumber = 1

#collision avoidance:
hColl = -10
rColl = 0

#in the original code the simulation runs it on 0 once and then 1 five times
distf = 1

#this determines the "characteristic length of signal decay",
#original code iterates through [1, 1, 2, 4, 8, 16]
adistf = 1

#step size
dt = 0.1

#v0: velocity I believe
v0 = 0.05

#v0t: velocity of targets I believe
#in the original code it is initialized to zero but that doesn't really make sense to me
v0t = np.zeros(ntargets)
for i in range(ntargets):
    v0t[i] = 0

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



# --- Details of the simulation to change ---

#allocentric flag
allocentricFlag = [0]

#attraction
h0s = [[0.45,0.46]]

#hbase
h_b = [0.2]

#width of the gauss bump 
sigma = np.linspace(0.1,0.7,num=100)

#noise parameter 
beta = [100]

# data we will collect each time we run the simulation
targets_reached = []
time_to_target = []
movement_starts = []
min_distance = []

decision1_time = []
decision2_time = []

allo = []
attraction = []
h_bs = []
sigmas = []

# number of times to run the simulation
n_samples = 1

# -------- Running the simulation --------

#this is running the sim with a bunch of changes in variables

for sim in range(n_samples):
    for orientation in range(len(allocentricFlag)):
        for h in range(len(h0s)):
            h0 = h0s[h]
            for hb in range(len(h_b)):
                for s in range(len(sigma)):
                    for b in range(len(beta)):
                        headings, xPos, yPos, targetsx, targetsy = simulate_ringattractor.simulate_ring_attractor(N,L,T,ntargets,nagents,allocentricFlag[orientation],periodicflag,rEgo,rEgoTarget,Egonumber,
                                    distf,adistf,J,beta[b],h0,h_b[hb],dt,v0,v0t,sigma[s],hColl,rColl,
                                    initialx,initialy,initialxt,initialyt,False,False)
                        
                        final_decision, time_reached, movement_start = simulation_metrics.get_destination_metrics(xPos,yPos,targetsx,targetsy)
                        decision_indices = simulation_metrics.get_direction_info(headings,movement_start,time_reached,1)
                        success_measure = simulation_metrics.get_min_distance(xPos, yPos, targetsx, targetsy, True, 1)
                        movement_starts.append(movement_start)
                        targets_reached.append(final_decision)
                        time_to_target.append(time_reached)
                        min_distance.append(success_measure)
                        allo.append(allocentricFlag[orientation])
                        attraction.append(h0s[h])
                        h_bs.append(h_b[hb])
                        sigmas.append(sigma[s])
                        if len(decision_indices) > 0:
                            decision1_time.append(decision_indices[0])
                        else:
                            decision1_time.append(-1)
                        print(f"time: {s}")

sim_data = {'Distance to closest target over sum of distances': min_distance,'Final target': targets_reached,'Time to target': time_to_target, 'Movement starts': movement_starts, 'First decision time': decision1_time, 'Allocentric or egocentric': allo, 'Attraction': attraction, 'Base attraction': h_bs, 'sigma': sigmas}
sim_df = pd.DataFrame(sim_data)
pd.set_option('display.max_columns', None)
sim_df.to_csv("simulation_resuts.csv") 

# plot x axis as param of interest, plot y axis as success measure
plt.figure(1)
plt.plot(sigma, min_distance)
plt.title("Success over different sigma values (uneven attraction, egocentric)")
plt.xlabel("Sigma")
plt.ylabel("Smallest distance to better target (over sum of distances)")
plt.show()
