%tafavot ba v3: both speed and direction are calcualted based on neuron activity

%Az  simulateRingAttractorFlockingv3v2TargetsreanchoringtimeRand copy
%shode. Switch distance-dependence ast.

function [x, targetpos, uArray, headings] = ...
    simulateRingAttractorFlockingv3v2TargetsreanchoringDistanceNumb( ...
        numAgents, numTargets, N, T, dt, v0, v0t, L, ...
        W, h_b, beta, ...
        h0, hColl, sigma, rColl, ...
        initialXY, initialTargetPositions, ...
        periodicFlag, ...
        allocentricFlag,distf,adistf,rEgo,rEgoTarget,Egonumber)
% SIMULATERINGATTRACTORFLOCKINGv3
% --------------------------------------------------------
% This function implements a ring-attractor model with continuous 
% Amari-style dynamics, controlling each agent's heading. It is extended
% to include 'numTargets' targets performing random walks, which also 
% provide Gaussian bumps to each agent's ring. 
%
% ALLOCENTRIC vs. EGOCENTRIC: (unchanged except for minimal additions)
%  - If allocentricFlag=1 => The ring angles remain in absolute coords
%  - If allocentricFlag=0 => ring angles rotate with agent's heading
%
% usage:
% [xPos, yPos, targetXPos, targetYPos, uArray, headings] = ...
%   simulateRingAttractorFlockingv3( ...
%       numAgents, numTargets, N, T, dt, v0, v0t, L, ...
%       W, h_b, beta, h0, hColl, sigma, rColl, ...
%       initialXY, initialTargetPositions, periodicFlag, allocentricFlag );
%
%    * xPos, yPos => (numAgents x T+1) positions of agents
%    * targetXPos, targetYPos => (numTargets x T+1) positions of targets
%    * uArray     => (N x numAgents x T+1) ring states
%    * headings   => (numAgents x T+1) agent headings
%
% Explanation of key steps (same as before), plus:
%  1) We also do a simple random walk for each target:
%       targetXPos(i,t+1) = targetXPos(i,t) + v0t(i)*sign(randn);
%       targetYPos(i,t+1) = targetYPos(i,t) + v0t(i)*sign(randn);
%  2) Each target adds a Gaussian bump to the ring activity of each agent.

%% 0) Initialization

% Each agent has its own ring angles alphaRing(a,:) in [0..2*pi)
alphaRing0 = linspace(0,2*pi,N+1);
alphaRing0 = alphaRing0(1:end-1); 
alphaRing = zeros(numAgents, N);  % (a,i)
for a=1:numAgents
    alphaRing(a,:) = alphaRing0;
end

fAct = @(u)(tanh(beta*u));  % activation function
fAct = @(u)((1+tanh(beta*u))/2);
% Store ring states: (N x numAgents x T+1)
uArray = zeros(N, numAgents, T+1);
u0 = 0.1 * randn(N, numAgents);
uArray(:,:,1) = u0;

% Store agent positions, headings
xPos = zeros(numAgents, T+1);
yPos = zeros(numAgents, T+1);
headings = zeros(numAgents, T+1);

for a = 1:numAgents
    xPos(a,1) = initialXY(1,a);
    yPos(a,1) = initialXY(2,a);
    headings(a,1) = 2*pi*rand; % random heading
    % SHIFT ring by that heading if allocentricFlag=0 => we do it later
    % but you already do it in code, so keep it:
    alphaRing(a,:) = mod(alphaRing(a,:) + headings(a,1), 2*pi); %does this matter if not allocentric?
end

% Store target positions
targetXPos = zeros(numTargets, T+1);
targetYPos = zeros(numTargets, T+1);
for ti = 1:numTargets
    targetXPos(ti,1) = initialTargetPositions(ti,1);
    targetYPos(ti,1) = initialTargetPositions(ti,2);
end


%% 1) Time loop
for tStep = 1:T

    %% A) Build external field from neighbors and from targets
    Iextern = zeros(N,numAgents);

    Egocentric=zeros(numAgents,1);
    % 1) Contribution of other agents
    for a=1:numAgents
        xa = xPos(a,tStep);
        ya = yPos(a,tStep);

        % Compare with each other agent b
        for b=1:numAgents
            if b==a, continue; end
            xb = xPos(b,tStep);
            yb = yPos(b,tStep);

            dx = xb - xa;
            dy = yb - ya;
            if periodicFlag==1 %a little confused about this... (seems like not a big deal though)
                % wrap
                if abs(dx)>L/2, dx=dx - sign(dx)*L; end
                if abs(dy)>L/2, dy=dy - sign(dy)*L; end
            end
            distAB = sqrt(dx^2 + dy^2);

            % amplitude
            ampl = h0(numTargets+b);
            if distf~=0;ampl=ampl*(exp(-adistf*distAB/L));end
            if distAB < rColl
                ampl = hColl;  % negative => collision avoidance
            end
            if distAB < rEgo
                Egocentric(a)=Egocentric(a)+1;
            end

            angleAB = atan2(dy, dx);
            if angleAB<0, angleAB = angleAB + 2*pi; end

            for i=1:N
                dAng = abs(alphaRing(a,i) - angleAB);
                if dAng>pi, dAng = 2*pi - dAng; end
                Iextern(i,a) = Iextern(i,a) + ampl*exp(-0.5*(dAng^2)/(sigma^2));
            end
        end
    end

    % 2) Contribution of targets
    for a=1:numAgents
        xa = xPos(a,tStep);
        ya = yPos(a,tStep);

        for ttarg = 1:numTargets
            xt = targetXPos(ttarg,tStep);
            yt = targetYPos(ttarg,tStep);

            dx = xt - xa;
            dy = yt - ya;
            if periodicFlag==1
                if abs(dx)>L/2, dx=dx - sign(dx)*L; end
                if abs(dy)>L/2, dy=dy - sign(dy)*L; end
            end
            distAB = sqrt(dx^2 + dy^2);
            if distAB < rEgoTarget
                Egocentric(a)=Egocentric(a)+1;
            end

            % same amplitude logic: No collision avoidance with targets
            ampl = h0(ttarg);
            if distf~=0;ampl=ampl*(exp(-adistf*distAB/L));end
%             if distAB < rColl
%                 ampl = hColl; 
%             end

            angleAB = atan2(dy, dx);
            if angleAB<0, angleAB=angleAB+2*pi; end

            for i=1:N
                dAng = abs(alphaRing(a,i) - angleAB);
                if dAng>pi, dAng = 2*pi - dAng; end
                Iextern(i,a) = Iextern(i,a) + ampl*exp(-0.5*(dAng^2)/(sigma^2));
            end
        end
    end

    %% B) Update ring states via Euler method
    for a=1:numAgents
        uaOld = uArray(:,a,tStep);
        faOld = fAct(uaOld);

        netRing = (1/N)*( W(:,:,a)*faOld );
        dU = -uaOld + netRing - h_b + Iextern(:,a);
        uaNew = uaOld + dt*dU;
        uArray(:,a,tStep+1) = uaNew;
    end

    %% C) Compute heading from ring
    for a=1:numAgents
        uaNow = uArray(:,a,tStep+1);
        faNow = fAct(uaNow);
        faPos = faNow;
        faPos(faPos<0)=0;  % only positive part

        cx=0; cy=0;
        for i=1:N
            cx = cx + faPos(i)*cos(alphaRing(a,i)); %read more in the paper about why this works for cx and cy?
            cy = cy + faPos(i)*sin(alphaRing(a,i));
        end
        newAngle = 0;
        if abs(cx)<1e-9 && abs(cy)<1e-9
            newAngle = headings(a,tStep);  % fallback
        else
            newAngle = atan2(cy, cx);
            if newAngle<0, newAngle=newAngle+2*pi; end
        end
        headings(a,tStep+1) = newAngle;

        
        % If EGOCENTRIC => rotate ring so alphaRing(a,i) remains "agent-based"
        if allocentricFlag==0
            alphaRing(a,:) = mod(alphaRing(a,:) - headings(a,tStep) + newAngle,2*pi);
        elseif Egocentric(a)>=Egonumber
            alphaRing(a,:) = mod(alphaRing(a,:) - headings(a,tStep) + newAngle,2*pi);
        end
    end

        %% D) Move each agent
    for a = 1:numAgents
        oldx = xPos(a,tStep);
        oldy = yPos(a,tStep);
        % Compute the weighted sum of contributions from each neuron.
        cx = 0; 
        cy = 0;
        for i = 1:N
            % Use the positive part of the activation:
            val = fAct(uArray(i,a,tStep+1));
            if val < 0
                val = 0;
            end
            cx = cx + val * cos(alphaRing(a,i));
            cy = cy + val * sin(alphaRing(a,i));
        end
        % Update positions using the net contribution.
        newx = oldx + dt * v0 * cx;
        newy = oldy + dt * v0 * cy;
        
        if periodicFlag == 1
            newx = mod(newx,L);
            newy = mod(newy,L);
        else
            if newx < 0, newx = 0; elseif newx > L, newx = L; end
            if newy < 0, newy = 0; elseif newy > L, newy = L; end
        end
        
        xPos(a,tStep+1) = newx;
        yPos(a,tStep+1) = newy;
    end


    %% E) Move each target with random walk (like your Hopfield code)
    for ttarg=1:numTargets
        oldXT = targetXPos(ttarg,tStep);
        oldYT = targetYPos(ttarg,tStep);

        % random +/- step of size v0t(ttarg)
        newXT = oldXT + v0t(ttarg)*sign(randn(1));
        newYT = oldYT + v0t(ttarg)*sign(randn(1));

        if periodicFlag==1
            newXT = mod(newXT,L);
            newYT = mod(newYT,L);
        else
            if newXT<0, newXT=0; elseif newXT>L, newXT=L; end
            if newYT<0, newYT=0; elseif newYT>L, newYT=L; end
        end

        targetXPos(ttarg,tStep+1) = newXT;
        targetYPos(ttarg,tStep+1) = newYT;
    end

    %% Visualization (optional)
    if mod(tStep,100)==0
        figure(1)
        figure(1); clf; hold on;
        % Plot agents
        if numAgents>0
            scatter(xPos(:,tStep+1), yPos(:,tStep+1), 10, 'filled','MarkerFaceColor',[0,0.2,0.8]);
        end
        % Plot targets
        if numTargets>0
            scatter(targetXPos(:,tStep+1), targetYPos(:,tStep+1), 10, 's','filled','MarkerFaceColor',[0.8,0,0.2]);
        end
        axis([0 L 0 L]); axis square;
       %axis([min(xPos(:,tStep+1)) max(xPos(:,tStep+1)) min(yPos(:,tStep+1)) max(yPos(:,tStep+1))]); axis square;
        title(sprintf('Time step %d (beta=%.2f)', tStep, beta));
        drawnow;

                figure(2);
agentIDs = [1];  % we choose agents 1..4 for example
for sp = 1:length(agentIDs)
    subplot(2,2,sp);
    % final ring states of the chosen agent
    aID   = agentIDs(sp);
    uFin  = uArray(:, aID, tStep+1);
    fFin  = fAct(uFin);
    plot(alphaRing(sp,:), fFin, 'b.-');
    hold on; plot([0,2*pi],[0,0],'k--');
    title(sprintf('Agent %d ring final', aID));
    xlabel('Ring angle'); ylabel('f(u)');
    ylim([-1.1,1.1]);
    hold off
  %  pause(0.1)
end

    end
end

x(1:2:2*numAgents-1,:)=xPos;
x(2:2:2*numAgents,:)=yPos;
targetpos(:,1,:)=targetXPos;
targetpos(:,2,:)=targetYPos;
end
