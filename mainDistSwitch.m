


clear; clc;

%baraye 2 target
clear

numcpu=5;

l2=1;
nagent=1*l2;

hbase=[0.2]/l2;

beta=[10];

%number of targets
v0t=zeros(104,1);L=100;
ntarget=length(v0t);
% Initial positions of targets
% must be numTargets x 2
initialtarget = zeros(2,ntarget);
%initialtarget(1,:) = [0.25*L, 0.70*L];
%initialtarget(2,:) = [0.75*L, 0.70*L];

% for i=1:ntarget
% initialtarget(i,1) = L/4+L/2*rand(1);
% initialtarget(i,2) =  L/2+(L/4)*rand(1);
% end


%% --- define your sub-area and circle ---
x0 = L/4;   x1 = 3*L/4;
y0 = L/2;   y1 = 3*L/4;
xc = (x0 + x1)/2;    % circle center X
yc = (y0 + y1)/2;    % circle center Y
R  = min(x1-x0, y1-y0)/2;  % so circle just fits in the rectangle

% --- define your circular region (center + radius) ---
x0 = L/4;   x1 = 3*L/4;
y0 = L/2;   y1 = 3*L/4;
xc = (x0 + x1)/2;      % circle center X
yc = (y0 + y1)/2;      % circle center Y
R  = min(x1-x0, y1-y0)/2;  % radius so circle just fits in rect

% --- choose a grid-spacing s so that density � ntarget ---
s = sqrt(pi*R^2/ntarget);   % area/RoughCount ? spacing
% (you can tweak this constant if you want more or fewer total points)

%% --- build full square grid over [xc-R,xc+R]�[yc-R,yc+R] ---
xs = xc - R : s : xc + R;
ys = yc - R : s : yc + R;
[GX, GY] = meshgrid(xs, ys);

% --- mask out the points outside the circle ---
inCircle = (GX - xc).^2 + (GY - yc).^2 <= R^2;
pts      = [GX(inCircle), GY(inCircle)];   % Npts�2

% --- pack into initialtarget ---
initialtarget = pts';        % 2�Npts
ntarget       = size(initialtarget, 2);


%attraction
h00s=0.1;

h00=0.1*ones(ntarget,1);
h00(20)=1;

for j=1:size(h00,2)
    for k=1:ntarget
        h0(k,j)=h00(k,1);
    end

for k=1:nagent
    h0(ntarget+k,j)=h00s(j);
end
end


Nspace = 0.1;     % width of the gauss bump

%h to be a function of distance: distf=0 means no distance dependence and
%adistef determines the characteristic length of signal decay (taken to be
%an exponential)
distf =[0 1 1 1 1 1];
adistf=[1 1 2 4 8 16];
Ni=length(hbase);
Nj=length(distf);
Nk=length(Nspace);
Nr=5;

N=100;

T=30000;
dt = 0.1;        % integration step for ring, also for movement
v0d=.05;


periodicspace=0;

initposnoise=1;
for r=1:Nr
for ng=1:nagent
initialx(r,1,ng)=L/2;L*rand(1); %a little confused on what this is meant to do
initialx(r,2,ng)=1*L/10;L*rand(1);
end
end
% 
%  MyCluster = parcluster
%  MyCluster.NumWorkers=numcpu
%  parpool(numcpu)
% 
% 


nu=0.5;

teta=linspace(0,2*pi,N+1);
teta=teta(1:end-1);
for i=1:N
J(i,:)= cos(pi*(abs(teta-teta(i))/pi).^nu);

deltah=abs(teta-teta(i));
deltah=pi-abs(pi-deltah);
J(i,:)=cos(pi*(deltah/pi).^nu);
J(i,i)=0;
end

J=squeeze(J);

%set equal to zero
collisionzone=[0];
hcoll=-10;


%always zero
allocentricFlag = 0;  

rEgo=0;
rEgoTarget=[0 0.5 1 2 4];
Egonumber=1;
Nk=length(rEgoTarget);

%% 4) Run the simulation
for k=1:Nk
    k
    for i=1:Ni
    i
        for j=1:Nj
            for r=1:Nr

            %[x(:,:,i,j,k,r), target(:,:,:,i,j,k,r), uArray(:,:,:,i,j,k,r), headings(:,:,i,j,k,r)] = ...
            [x(:,:,i,j,k,r)] = ...
            simulateRingAttractorFlockingv3v2TargetsreanchoringDistanceNumb( ...
        nagent, ntarget, ...
        N, T, dt, v0d, v0t, L, ...
        J(:,:,1:nagent), hbase(i), beta, ...
        h0(:,1), hcoll, Nspace, collisionzone, ...
        squeeze(initialx(r,:,:))', initialtarget', ...
        periodicspace, ...
        allocentricFlag,distf(j),adistf(j),rEgo,rEgoTarget(k),Egonumber);
        

            end
        end
    end
end

save('2tN400')
