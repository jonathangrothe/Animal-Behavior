Here are some of the previous simulation settings so they don't bog down the code doc:

Basic unchanged params:
N = 100
L = 100
T = 5000 (with breaking at target enabled)
periodicflag = 0
rEgo = 0 
rEgotarget = 0
Egonumber = 1
hColl = -10
rColl = 0
distf = 1
adistf = 1
dt = 0.1
v0 = 0.05
v0t = 0*ntargets
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
nagents = 1
allocentricflag = [0,1]
h_b = 0.2
beta = 100

sim 1: repulsion near a more attractive target
ntargets = 4
initialx = L/2 - 30
initialy = L/2
initialxt = [L-15, L-15, L-15, L-15]
initialyt = [L/2+30, L/2-30, L/2+20, L/2-20]
h0s = [[0.47,0.47,-0.05,0.01,0], [0.67,0.47,-0.2,0.01,0]]
sigma = 0.5

sim 2: uneven attractions where ego roams, allo goes to target
ntargets = 4
initialx = 5
initialy = 5
initialxt = [12, 45, 75, 70]
initialyt = [70, 75, 45, 10]
h0s = [[0.4,0.5,0.6,0.2,0]] 
sigma = [0.6]

sim 3: 'Leading' target in otherwise symmetric geometry influences allo, freezes ego
ntargets = 5
initialx = 5
initialy = 5
initialxt = [10, 45, 75, 70, 95]
initialyt = [70, 75, 45, 10, 85]
h0s = [[0.85,0.85,0.85,0.85,0.15]]
sigma = 0.5

