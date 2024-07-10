clear all
% Written by Arno Gekiere during 09/2019
% Everything is based on the Marginal Value Theorem (MVT).

% This file is essentially a tutorial/explanation for foraging and MVT. I
% initially built it to develop my own understanding of MVT and foraging.

% In addition to this, All of the principles in this file are implemented
% in the two class scripts Patch.m and Environment.m. These scripts are
% meant to simplify the process of testing out different variables for
% environments and patches. 
% The "MVT_functions_Arno" file demonstrates how to use these class scripts
% to create any environment with any patch combinations using variables of
% your choice.

% Foraging happens in patches, each patch has its own amount of resources.
% As you are foraging in a patch the resources in that patch decrease over
% time. Subsequently the rate at which you collect these resources
% decreases until there are no resources left. (We will refer to collecting
% resources as gaining reward.) you will effectively be gaining less reward
% the longer you stay in a patch, as resources decrease. In other words
% staying in a patch is costly (patch residence time; PRT). So past a
% certain point the current reward rate will no longer be optimal when
% compared to what one could gain in another patch. So essentially there is
% an optimal point in PRT that dictates when you should move to another
% patch to maximize reward rate.

% Decay value, how fast the resources (energy) depletes in a patch.
%a = 0.11;
a = 0.075; 

% If every patch were equal in starting resources and if moving to another
% patch could be instantaneous then the optimal time to leave a patch would
% be immediately after collecting one unit of resource and moving to one of
% the next patches as the only cost is staying in a patch. This is because
% the rate at which you gain rewards is maximized this way.

% However, when traveling to another patch you do not arrive there
% instantaneously instead traveling to another patch takes a set amount of
% time, the travel time. The travel time is essentally a cost, the cost to
% move from one patch to another patch. This makes it so that you have to
% gain enough reward and thus stay longer in a patch to overcome the cost of
% traveling to another patch before you reach the maximum reward rate.
travel_t = 6;

% Additionally, not every patch has an equal amount of resources to start
% with in this case we use low quality patches and high quality patches.
% Respectively, patches with a lower amount and patches with a higher
% amount of starting resources. We also assume that both type of patches
% occur with a different probability. So an environment can for example
% have more high quality patches than low quality patches, that would be
% considered a rich environment. It is also not known beforehand at which
% type of patch you will arrive when deciding to move to another patch.
% This makes it so that the optimal time to move to a new patch is when the
% resources of the current patch reach below the average of all available
% patches. Since the initial resources of a quality patch are closer to the
% average resources of all patches, the optimal leaving time will be
% shorter than a high quality patch.

% the initial reward rate upon entering a fresh patch. In this case there
% are two types of patches: low quality (34.5) and high quality (57.5).
patch_start_R = [32.5 45 57.5];
%patch_start_R = [34.5 57.5];
A = patch_start_R;

% Number of patch types:
patch_n = size(patch_start_R, 2);

% To simulate all this during the examples we also create two separate
% environments. (1) An environment where the high quality patch occurs more
% frequently than the low quality patch (rich env). And (2) an environment
% where the low quality patch is more dominant (poor env).
% In order to reflect this, we have a higher probability of high quality
% patches in a rich environment: 0.7 as opposed to the probability of high
% quality patches in a poor environment: 0.3
%EXAMPLE
% rich environment -> [0.3, 0.7] = [probability low quality, probability high quality]
% poor environment -> [0.7, 0.3] = [probability low quality, probability high quality]
% put together -> [0.3, 0.7; 0.7, 0.3] = [rich environment; poor environment]

% actual values, no longer example:
%env = [0.2 0.3 0.5; 0.7 0.15 0.15];
%env = [0.3 0.7];
env = [0.2 0.3 0.5; 0.5 0.3 0.2];
env_n = size(env,1);

% let's also set a time period over which we want to run the foraging.
measurements = 100000;
time = 100;
reso_fix = measurements/time;

% a thousand points between 1 and 50. So it is over 50 seconds but we make a 1000 measurements in that timeframe 
reso = linspace(1,time,measurements);

% How do we determine the optimal leaving time (OLT)?
% First we need the function that describes a reward rate that encompasses
% both separate reward rates while taking into account their probability of
% encountering. This is just the average reward rate in the environment:
%
%                R = (?(pi * patch_gain_total(t)i) / (d + ?(pi * ti))
%
% NOTE: 'i' in this case represents the current patch type.
%
% The function in code is defined at the bottom of the script. 
%
% Essentially there are five parameters:
%   - t: patch residency time
%   - p: probability of encountering patch type
%   - d: travel time between patches
%   - patch_gain_total(t): foreground reward rate
%   - k: energetic costs of foraging (currently excluded)
%
% The patch_gain_total function is defined at the bottom of the script.

% Now calculate the reward rate for the both patch types at each time
% point. Once we have the reward rate of a patch we can calculate the
% optimal leaving time for that patch.

%% Example 1
% For now both patch types are treated as being 100% probable in their own
% environment. So we are not yet taking into account probabilities of rich
% vs poor environments. We just create to reward rates for each patch type
% as if we are in an environment where only one patch type occurs.

E = zeros(patch_n, size(reso, 2));
total_E = zeros(patch_n, size(reso, 2));
instant_R = zeros(patch_n, size(reso, 2));
RR = zeros(patch_n, size(reso, 2));
for type = 1:patch_n
    % for t = 1:reso
    for t = 1:size(reso, 2)
        % Fill an with the current energy gain at each timepoint, this
        % energy decays as resources deplete. The function "patch_gain"
        % contains the math function to determine these values.
        E(type, t) = patch_gain(reso(t), patch_start_R(type));
        
        % Then fill an array with the energy gained so far at each timepoint
        % so from all energy gains summed up from timepoint 1 through 't'
        % option 1: just sum up E
        %total_E(type, t) = sum(E(type, 1:t))/reso_fix;
        
        % option 2: using "geometric series" (wiki this if you don't know
        % what it is) we can essentially use a math function for summing up
        % the E. Look inside "patch_gain_total" for the actual math
        total_E(type, t) = patch_gain_total(reso(t), patch_start_R(type));
        
        % Get the slope of the currently gained reward to get our
        % instantaneous reward (foreground reward).
        instant_R(type, t) = der_patch_gain(reso(t), patch_start_R(type));
        
        % The reward rate (RR) is simply total energy gain divided by time
        % spent in patch+travel time. (the function described in the
        % previous section above, except we don't deal with probabilities
        % in this example)
        RR(type, t) = total_E(type, t)/(travel_t + reso(t));
    end
end


close all % close open plots if any

% plot
figure() % create new figure to plot on
hold on
% Let's plot our RR for all patch types
for i = 1:size(RR, 1)
    plots(i) = plot(RR(i,:)), xlabel('Patch Residence Time (t)'), ylabel('RR'), title('')
    y_line(i) = yline(max(RR(i,:)));
    y_line(i).Color = plots(i).Color;
end

% Calculate OLT for each patch and put in array: the optimal leaving time
% simply coincides with the maximimzed point on the RR curve plotted above.
for i = 1:size(RR, 1)
    OLT(i) = find(max(RR(i,:))==RR(i,:)); % OLT for current patch in loop
    maxRR(i) = max(RR(i,:));
end

figure() % create new figure to plot on
hold on
% Let's plot the foreground instantaneous reward for all patches, the RR
% and the OLTs. These should a connect in 1 point. The optimal point to
% leave the patch. (you may need to increase the resolution for this to be
% exact as we are dealing with discrete values to draw these functions. To
% increase resolution increase the measurements variable on top to 10 000.
for i = 1:size(instant_R, 1)
    plots(i) = plot(instant_R(i,:)), xlabel('Patch Residence Time (t)'), ylabel('Foreground'), title('');
    plot(RR(i,:))
end

% Because we are using the same exponential function to decline the rewards
% gained for both patches and each patch is regarded as being 100% probable
% in their own environment, they will both share the same OLT.
for i = 1:size(instant_R, 1)
    x1 = xline(OLT(i));
    x1.Color = plots(i).Color;
end



%% Example 2
% In the previous example the patches were looked at seperately as being in
% their own environment where only that type of patch occurs.

% Now we will use the two seperate environments, an environment where the
% high quality patch occurs more frequently than the low quality patch
% (rich env). And an environment where the low quality patch is more
% dominant (poor env). We already created these environments earlier in the
% the 'env' variable.

% Before, in example 1 we calculated two separate reward rates, one for
% each patch type. Each patch type considered to be 100% probable in their
% own environment. For this example both patch types can occur in the same
% environment with a certain probability to encountering each patch type.
% This means that we have to calculate a reward rate that accounts for
% this. We essentially need a reward rate which encompasses both separate 
% reward rates while taking into account their probability of encountering.
% Which is just the average reward rate in the environment (function
% described in the intro).

% Create a reward rate of the environment array to hold our values
%RRE = zeros(2, reso);
RRE = zeros(env_n, size(reso, 2));
RRE_patch = zeros(env_n, size(reso, 2), patch_n);

for env_type = 1:env_n % this loop is to create an RRE for both environments

    for p_type = 1:patch_n
        % for t = 1:reso
        
        
        for t = 1:size(reso, 2)
            % The environment reward rate (RRE) is simply total energy gain
            % divided by time spent in patch+travel time while taking into
            % account the probability of each patch type occurring.
            %R = (env(env_type, p_type) * total_E(p_type, t))/(travel_t + env(env_type, p_type) * reso(t))
            %RRE(env_type, t) = RRE(env_type, t) + ((env(env_type, p_type) * total_E(p_type, t))/(travel_t + env(env_type, p_type) * reso(t)));
            RRE(env_type, t) = RRE(env_type, t) + env(env_type, p_type) * (total_E(p_type, t)/(travel_t + reso(t)));
            RRE_patch(env_type, t, p_type) = RRE_patch(env_type, t,p_type) + env(env_type, p_type) * (total_E(p_type, t)/(travel_t + reso(t)));
        end
    end
end


% get the RRE
for i = 1:size(RRE, 1)
    maxRRE(i) = max(RRE(i,:)); % get the RRE at which it is optimal to leave
  
end

%OLT1 = find(RRleave1==RRE(1,:)); % OLT for rich environment
%OLT2 = find(RRleave2==RRE(2,:)); % OLT for poor environment

%close all % close open plots if any

% Let's plot our RR for all our environments
figure() % create new figure to plot on
hold on
for i = 1:size(RRE, 1)
    plots(i) = plot(RRE(i,:)), xlabel('Patch Residence Time (t)'), ylabel('RR'), title('RR for environment (background RR)');
    for p_type = 1:patch_n
        plot(RRE_patch(i,:,p_type)), xlabel('Patch Residence Time (t)'), ylabel('RR'), title('RR for environment (background RR)');
        OLT2(i, p_type) = optimal_lt(patch_start_R(p_type), maxRRE(i));
    end
    y_line(i) = yline(maxRRE(i));
    y_line(i).Color = plots(i).Color;
end

% Let's plot our RR for all patch types on there too without probabilities
for i = 1:size(RR, 1)
    plot(RR(i,:)), xlabel('Patch Residence Time (t)'), ylabel('RR'), title('')
end


% Overlap RRE and instantaneous R
figure() % create new figure to plot on
hold on
for i = 1:size(RRE, 1)
    plots(i) = plot(RRE(i,:)), xlabel('Patch Residence Time (t)'), ylabel('RR'), title('RR for environment (background RR)');
end

for i = 1:size(instant_R, 1)
    plots(i) = plot(instant_R(i,:)), xlabel('Patch Residence Time (t)'), ylabel('Foreground'), title('');
end

% Draw the lines to indicate RRE leave
for i = 1:size(maxRRE, 2)
    y_line(i) = yline(maxRRE(i));
    y_line(i).Color = plots(i).Color;
end

figure() % create new figure to plot on
hold on
% Let's plot the foreground instantaneous reward for all patches
for i = 1:size(instant_R, 1)
    plots(i) = plot(instant_R(i,:)), xlabel('Patch Residence Time (t)'), ylabel('Foreground'), title('');
end

% Draw the lines to indicate
for i = 1:size(maxRRE, 2)
    y_line(i) = yline(maxRRE(i));
    y_line(i).Color = plots(i).Color;
end


% patch gain = energy you gain in patch at single timepoint t
function output = patch_gain(t, A)
    output = A * exp(-0.075*t);
end

% total energy gained so far up to time point t
function output = patch_gain_total(t, A)
    a = A;
    r = exp(-0.075);
    r_t = exp(1)^-0.075^t;
    output = a*(1-r_t)/(1-r);
    %output =(A/0.1)*(1-exp(-0.1*t));
end

% foreground reward rate = derivative of patch gain
function output = der_patch_gain(t, A)
    r = exp(-0.075);
    r_t = exp(1)^-0.075^t; % r^t
    output = (A * r_t * log(r))/(r-1);
end

% environmental reward rate
function output = reward_rate(p, t, g, d)
    output = (p * g) / (d + p * t);
end

function output = der_reward_rate(p1, p2, a1, a2, t, A, tr)
    r = exp(-0.075);
    r_t = exp(1)^-0.075^t; % r^t
    output = ((a2*p2+a1*p1)*(r_t(ln(r)*(t+tr)-1)+1))/((r-1)*(t+tr)^2)
end

function output = optimal_lt(A, y)
    r = exp(-0.075);
    output = log(y*(r-1)/(A*log(r)))/log(r);
end
