clear all
%% Make example MVT curves here with different travel times, decay rates of the exponential and scaling factors.
% Written By Nils Kolling 3rd of June 2015 For Illustration
% You can easily add other functions. However, remember that they need to
% fulfill certain criteria to for this solution to be an optimal one,
% Although you can always just use this to calculate the optimal RR by
% brute force, in which case you need not worry about those conditions.
% In other words, even if your curve is complex, non-MVT like, there will always be an
% average RR that is the highest given your parameters and delays.

%% Set Some Parameters here %%%%%%%%%%%%
TravelTs=[6]; %NK: This isn't necessarily seconds but relative to how you scale the rest.
reso=50;   %NK: Actually its length of the time, but turns into resolution if you change the time scaling accordingly.
A=[32.5 45 57.5]; % initial yield of the 3 patch types (low medium high)
a=[.075 .075 .075]; % decay rate of the 3 patch types (low medium high). All decay at the same rate.

pPatch = [2/10 3/10 5/10]; % rich environment - proportion of low yield, mid yield, and high yield patches
% pPatch = [5/10 3/10 2/10]; % poor environment  - proportion of low yield, mid yield, and high yield patches

NrPatchTypes=length(a);
%% Calculate important quantities %%%%%

AllRewardE = zeros(numel(reso), NrPatchTypes);
AllRRE=zeros(numel(reso), NrPatchTypes);
AllGainE=zeros(numel(reso), NrPatchTypes);

for P=1:NrPatchTypes
    for T=1:reso
        RewardE(T) = A(P)* exp(-a(P)*T) ;      % Your exponential function
        %RewardL(T) = A(P)-a(P)*T;          % If you want a linear function
        RRE(T)=-a(P)*A(P)*exp(-a(P)*T);           % The first derivative i.e. reward rate for the exponential
        %RRL(T) = -a(P)*A(P)-a(P)*T;

        % exponential
        GainE(T)=(A(P)/a(P))*(1-exp(-a(P)*T));
        AllRewardE(T,P) = A(P)* exp(-a(P)*T) ;      
        AllRRE(T,P)=-a(P)*A(P)*exp(-a(P)*T);
        AllGainE(T,P)=(A(P)/a(P))*(1-exp(-a(P)*T));

    end

    % As the optimal leaving time depends on travelling time here is a
    % brute force method to calculate the optimal leaving time for a series
    % of Travel Times. (Because we know the function is decelerating we could find the point at which the derivative of the RR is 0,
    % but that ends up being harder, because you need to do quite a bit of math)

    Tind=0;
    for TravelT=TravelTs % for each travel time
        Tind=Tind+1;
        for T=1:reso
            RR(T,Tind)=GainE(T)/(TravelT+T); % Just Total Reward/time costs at each time T
           %  RR(T,Tind)=GainL(T)/(TravelT+T); % Just Total Reward/time costs at each time T

            multiPatchRR(T,Tind,P)=RR(T,Tind); % RR in each patch type in a given environment
        end
        Tleave(Tind)=find(max(RR(:,Tind))==RR(:,Tind), 1, 'first'); % Which row has the maximum RR - each row = one sec
        RRleave(Tind)=max(RR(:,Tind)); % and what is the RR at this time?
    end
end

for     Tind=1:length(TravelTs)

    Timesteps=1:reso; % NK: Used to be T but should be reso (I think)
    Timesteps=Timesteps+TravelTs(Tind); %Timesteps is now the total time cost at each leaving time (row)
    T2=repmat(Timesteps,[reso 1 reso]); % creates a TxT matrix, repeated T times , with each column one time cost (ie the same)
    Timesteps=Timesteps';
    T1=repmat(Timesteps,[1 reso reso]); % creates the transposed TxT matrix, repeated T times, with each row one time cost (ie the same)
    Pong(1,1,:)=Timesteps; % array with each cell containing a unique timepoint
    T3=repmat(Pong,reso,reso);    % TxT matrix, repeated T times, of each unique timepoint
    %     proportionT1=T1./(T1+T2+T3);
    %     proportionT2=T2./(T1+T2+T3);
    %     proportionT3=T3./(T1+T2+T3);    %
    proportionT1=T1.*pPatch(1)./(T1.*pPatch(1)+T2.*pPatch(2)+T3.*pPatch(3));
    proportionT2=T2.*pPatch(2)./(T1.*pPatch(1)+T2.*pPatch(2)+T3.*pPatch(3));
    proportionT3=T3.*pPatch(3)./(T1.*pPatch(1)+T2.*pPatch(2)+T3.*pPatch(3));


    curPatchRR(:,:,:,1)=repmat(multiPatchRR(:,Tind,1),[1 reso reso ]);
    curPatchRR(:,:,:,2)=repmat(multiPatchRR(:,Tind,2)',[reso 1 reso ]);
    tem(1,1,:)=multiPatchRR(:,Tind,3);
    curPatchRR(:,:,:,3)=repmat(tem,[reso reso 1]); % creates the same structure as the proportions but with the RRs at each time point

    OverallRR(:,:,:,Tind)=[curPatchRR(:,:,:,1).*proportionT1 + curPatchRR(:,:,:,2).*proportionT2  ... .*pPatch(1) .*pPatch(2) .*pPatch(3)
        + curPatchRR(:,:,:,3).*proportionT3];

    maxRR(Tind)=max(max(max(OverallRR(:,:,:,Tind))));
    tem2=OverallRR(:,:,:,Tind);

    [Tmax(Tind,1),Tmax(Tind,2),Tmax(Tind,3)]=ind2sub(size(tem2),find(tem2(:)==maxRR(Tind)));
     GainEmax(Tind,:)=[AllGainE(Tmax(Tind,1),1) AllGainE(Tmax(Tind,2),2) AllGainE(Tmax(Tind,3),3)];
     RewardEmax(Tind,:)=[AllRewardE(Tmax(Tind,1),1) AllRewardE(Tmax(Tind,2),2) AllRewardE(Tmax(Tind,3),3)];

    % GainLmax(Tind,:)=[AllGainL(Tmax(Tind,1),1) AllGainL(Tmax(Tind,2),2) AllGainL(Tmax(Tind,3),3)];
    % RewardLmax(Tind,:)=[AllRewardL(Tmax(Tind,1),1) AllRewardL(Tmax(Tind,2),2) AllRewardL(Tmax(Tind,3),3)];
end
%keyboard

%% Plot Figures
% close all
% figure('color',[1 1 1],'name','Rewards by time and leaving Thresholds');subplot(2,1,1);plot(AllRewardE);hold on;plot(Tmax,RewardEmax,'.k','markersize',10);plot(repmat(maxRR,reso,1),'--k');legend('low','med','high')
% subplot(2,1,2);plot(AllGainE);hold on;plot(Tmax,GainEmax,'.k','markersize',10);%axis([1 100 1 100])
% plot(repmat(maxRR,reso,1).*repmat([1:reso]',1,length(maxRR)),'--'); %+ repmat(maxRR.*TravelTs,reso,1)

