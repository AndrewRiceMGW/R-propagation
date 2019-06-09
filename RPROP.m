clear all, close all, clc
Name = 'RHess'
% by Dave Touretzky (1st modified by Nikolay Nikolaev) (2nd Modification by
% Andrew Rice)
% https://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/matlab/
load sunspot.dat
year=sunspot(:,1); relNums=sunspot(:,2); %plot(year,relNums)
ynrmv=mean(relNums(:)); sigy=std(relNums(:));
nrmY=relNums; %nrmY=(relNums(:)-ynrmv)./sigy;
ymin=min(nrmY(:)); ymax=max(nrmY(:));
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);

% create a matrix of lagged values for a time series vector
Ss=relNums'; idim=10; % input dimension
odim=length(Ss)-idim; % output dimension

for i=1:odim
    y(i)=Ss(i+idim);
    for j=1:idim
        x(i,j) = Ss(i-j+idim); %x(i,idim-j+1) = Ss(i-j+idim);
    end
end
% Initial Paramaters
Patterns = x'; Desired = y; NHIDDENS = 5; prnout=Desired;
[NINPUTS,NPATS] = size(Patterns); [NOUTPUTS,NP] = size(Desired);
LearnRate = 1.0 ; Momentum = 0; DerivIncr = 0; deltaW1 = 0; deltaW2 = 0;
Inputs1 = [ones(1,NPATS); Patterns];
Weights1 = 0.5*(rand(NHIDDENS,1+NINPUTS)-0.5);
Weights2 = 0.5*(rand(1,1+NHIDDENS)-0.5);
Vweights1 = 0.5*(rand(NHIDDENS,1+NINPUTS)-0.5);
Vweights2 = 0.5*(rand(1,1+NHIDDENS)-0.5);
TSS_Limit = 0.02;
Weights = [reshape(Weights1, 1, []), reshape(Weights2, 1, [])];
Out = zeros(1,NPATS); 
                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%                   FEED FORWARD              %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                                    %%
for epoch = 1:200
    if epoch > 1;
        Beta = Error;       
    else
        Beta(1,1:size(Out,2)) = 1;
        Hessian(length(Weights), length(Weights)) = 1; 
    end
%     Beta(1,1:size(Out,2)) = 1;
    %%
    % Forward propagation
    NetIn1 = Weights1 * Inputs1;
    RNetIn1 = Vweights1 * Inputs1; % R{aj}
    Hidden=1-2./(exp(2*NetIn1)+1); 
    RHidden = Hidden .* (1 - Hidden) .* (RNetIn1); % R{zj} 
    Inputs2 = [ones(1,NPATS); Hidden];
    RInputs2 = [ones(1,NPATS); RHidden]; 
    NetIn2 = Weights2 * Inputs2;
    RNetIn2 = (Weights2 * RInputs2) + (Vweights2* Inputs2); % R{yk} 
    Out = NetIn2;  prnout=Out; ROut = RNetIn2; Rprnout = ROut; 
    Error = Desired - Out; RError = Desired - ROut; 
    TSS = sum(sum( Error.^2 )); RTSS = sum(sum( RError.^2 ))
                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
%%%%%%%%%%%%%%%%%%%%%%%%%%    BACKPROPAGATION  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                                    %%    
    Beta(1,1:size(Out,2)) = 1;
    RBeta = Error; 
    bperr = ( Weights2' * Beta );
    HiddenBeta = (1.0 - Hidden .^2 ) .* bperr(1:end-1,:);
    RPrime = (1 - Hidden.*Hidden);% g'(aj)
    RPrime1 = (1 - Hidden)*Hidden';% g'(aj)
    RPrime2 = -2.0 * Hidden .* RPrime; % g"(aj)
    

    Rbperr1 = Vweights2'* RBeta; %sum(vkj,&k)
    Rbperr2 = Weights2' * ROut; % sum(wkj, R{&k})
    Rbperr3 = ( Weights2' * RBeta );

    RHIDBeta1 = RPrime.* Rbperr3(1:end-1,:);
    RHIDBeta2 = RPrime2.* RNetIn1.*Rbperr3(1:end-1,:);
    RHIDBeta3 = RPrime.*Rbperr1(1:end-1,:);% g' * sum(vkj,&k)
    RHIDBeta4 =  RPrime.*(Weights2(:, 1:end-1)* Hidden.*(1 -Hidden).* ROut);%g' * sum(wkj, R{&k})
    RHiddenBeta = RHIDBeta2 + RHIDBeta3 + RHIDBeta3; 

    % ((1-(1/1+e^-x))* ((1/1+e^-x) * 1-(1/1+e^-x))) + (((1/1+e^-x)*
    % 1-(1/1+e^x))) = sigmoid(x) - 3(sigmoid(x))^2 + 2(sigmoid(x))^3;
    
      
                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
%%%%%%%%%%%%%%%%%%%%%%%%%%       JACOBIAN      %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                                    %%
     sz = size(Inputs1,2); jacobian = zeros(sz,length(Weights));
     G=zeros(sz,length(Weights)); dW1 = zeros(1, 55); dW2 = zeros(1,6); 
    for d = 1:size(Inputs1,2) % 278
        count = 0;
        for d1 = 1:size(Inputs1,1) % 11
            for d2 = 1:size(HiddenBeta,1) % 5
                count = count + 1;
                dW1(d,count) = Inputs1(d1,d) * HiddenBeta(d2,d);
            end
        end
        % dW2 update
        for h = 1 :size(Inputs2,1) 
            dW2(d,h) = Beta(1,d) * Inputs2(h,d);        
        end
        G= [dW2 dW1];
        jacobian(d,:) = G(d,:);      
    end
    RY = (Hidden.*(1 -Hidden).* ROut);
    RZ = RInputs2;
%     sz = size(Inputs1,2); Rjacobian = zeros(sz,length(Weights));
%     RG=zeros(sz,length(Weights)); dW1 = zeros(1, 55); dW2 = zeros(1,6); 
%     for d = 1:size(Inputs1,2) % 278
%         count = 0;
%         for d1 = 1:size(Inputs1,1) % 11
%             for d2 = 1:size(HiddenBeta,1) % 5
%                 count = count + 1;
%                 HW1(d,count) = Inputs1(d1,d) * RHiddenBeta(d2,d);
%             end
%         end
%         % dW2 update
%         for h = 1 :size(Inputs2,1) 
%             HW2(d,h) = ((Hidden(1,d) * RY(d,1)) + (RZ(h,d) * RHIDBeta1(h,d)));
%         end
%         RG= [HW2 HW1];
%         Rjacobian(d,:) = RG(d,:);      
%     end

                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%            APPROXIMATE HESSIAN            %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                                    %%

%     Hessian = jacobian' * jacobian ;  
%     HessianInverse = inv(Hessian + (0.01 * eye(size(jacobian,2))));
%     HessianInverse = (HessianInverse/278)/100;

     for d = 1:size(Inputs1,2) % 278
        count = 0;
        for d1 = 1:size(Inputs1,1) % 11
            for d2 = 1:size(HiddenBeta,1) % 5
                count = count + 1;
                HW1(d,count) = Inputs1(d1,d) * RHiddenBeta(d2,d);
            end
        end
     end

                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%               EXACT HESSIAN               %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                                    %%
                                    
    HW1 = Inputs1 * RHiddenBeta';
%     HW2 = (sum(Inputs2 * RY') + sum(RZ* RHIDBeta1')');
    HW2i = (Inputs2 * RY');
    HW2ii = ( RZ * RHIDBeta1');
    HW2iii = HW2i + HW2ii;
    HW2= sum(HW2iii, 2)';
    Hb1 = sum(RHiddenBeta, 1);
    Hb2 = sum(RY, 1); 
%     Hessian = [HW1(:)', Hb1, HW2(:)', Hb2];
    VtH = [HW2(:)', HW1(:)'];
    Hessian = zeros(size(Weights,2));
    for V = eye(size(Weights,2))
        Hessian(find(V),:)  = VtH;
    end
                                    
                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%       SECOND  PASS WITH HESSIAN      %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                                    %%  
    % Forward propagation
    
    NetIn1 = Weights1 * Inputs1;
    Hidden=1-2./(exp(2*NetIn1)+1); %Hidden = tanh( NetIn1 );
    Inputs2 = [ones(1,NPATS); Hidden];
    NetIn2 = Weights2 * Inputs2;
    Out = NetIn2;  prnout=Out;
    % Backward propagation of errors
    Error = Desired - Out;
    TSS = sum(sum( Error.^2 )); % sum(sum(E.*E));
    Beta = Error;
    bperr = ( Weights2' * Beta );
    HiddenBeta = (1.0 - Hidden .^2 ) .* bperr(1:end-1,:);

    dW2 = Beta * Inputs2';
    dW1 = HiddenBeta * Inputs1';

                                    %%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%       DELTA UPDATES WITH HESSIAN      %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
    %%
    for dw2Col = 1:size(dW2,2)%6
        for dw2Row = 1:size(dW2,1)%1
            deltaW2(dw2Row,dw2Col) = Momentum + dW2(dw2Row,dw2Col) * ...
                Hessian(dw2Col,dw2Col);
        end
    end
    Count = 0;
    for dw1Row = 1:size(dW1,1)%11
        for dw1Col = 1:size(dW1,2)%55  
            Count = Count + 1;
            deltaW1(dw1Row,dw1Col) = Momentum + dW1(dw1Row,dw1Col) * ...
                Hessian(dw2Row+Count, dw2Col+Count);
        end
    end



                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%             WEIGHT UPDATES             %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                                    %%    


    Weights1 = deltaW1 + Weights1;
    Weights2 = deltaW2 + Weights2;


                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%               PRINT ERROR              %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                                    %%
    
    fprintf('Epoch %3d:  Error = %f\n',epoch,TSS);
    if TSS < TSS_Limit, break, end
    
 end
plot(year(idim+1:288),Desired,year(idim+1:288),prnout)
title('Sunspot Data')

