close all
clear all

%% Fuzzy Q-Learning

gamma = 0.7;    % discount factor
eta = 0.5;    % parameter learning factor 
epsilon = 0.9;  % exploration probability (1-epsilon = exploit / epsilon = explore)

% states
for i=1:81
    state(i) = i;
end
% actions
action = [0.05,0,-0.05];
% initial Q matrix
q1 = zeros(length(state),length(action));
epoch = 500;     % maximum number of iterations
state_idx = zeros(epoch,1);  
action_idx = zeros(epoch,1);  
state_idx(1) = 15; % the initial state to begin from
ST = [];
FUZZY = [];
alpha_i = [];
Q1 = zeros(length(state),length(action));
matrix = zeros(epoch,3);
V_t = zeros(length(state),1);
act = zeros(epoch,1);
reward = zeros(epoch+1,1); 
r1 = zeros(epoch,1); 
r2 = zeros(epoch,1); 
Pblock1 = zeros(epoch+1,1); 
Pblock2 = zeros(epoch+1,1); 
Pblock1_nd = zeros(epoch+1,1); 
Pblock2_nd = zeros(epoch+1,1); 
Pblock11 = zeros(epoch+1,1); 
Pblock12 = zeros(epoch+1,1);
Pblock21 = zeros(epoch+1,1); 
Pblock22 = zeros(epoch+1,1); 
Pblock11_nd = zeros(epoch+1,1); 
Pblock12_nd = zeros(epoch+1,1);
Pblock21_nd = zeros(epoch+1,1); 
Pblock22_nd = zeros(epoch+1,1); 
BitRate1 = zeros(epoch,1); 
BitRate2 = zeros(epoch,1); 
Q_error1 = zeros(epoch,1);
constant1 = -247;
constant2 = 50;
constant3 = 3;
delta = zeros(epoch,1);
delta_T1_C1=0;
delta_T1_C2=0;
delta_T2_C1=0;
delta_T2_C2=0;

fismat_T1_C2 = readfis('Fuzzy2-Train');
fismat_T2_C1 = readfis('Fuzzy3-Train');
fismat_T2_C2 = readfis('Fuzzy4-Train');

a = 0.05;
b = 0.3;
c = 0.75; 
d = 1.15;
lambda_t1_c1 = (b-a).*rand(epoch,1) + a;
lambda_t1_c2 = (b-a).*rand(epoch,1) + a;
lambda_t2_c1 = (d-c).*rand(epoch,1) + c;
lambda_t2_c2 = (d-c).*rand(epoch,1) + c;

trimf_1 = [-0.2 0.2 0.6];
trapmf_1 = [-1.2 -0.4 -0.2 0.2];
trapmf_2 =  [0.2 0.6 0.8 1.6];

mean11 = 10;
mean12 = 30;
mean13 = 50;
mean21 = 10;
mean22 = 30;
mean23 = 50;
mean31 = 10;
mean32 = 30;
mean33 = 50;
mean41 = 10;
mean42 = 30;
mean43 = 50;
sigma11 = 6;
sigma12 = 11;
sigma13 = 6;
sigma21 = 6;
sigma22 = 11;
sigma23 = 6;
sigma31 = 6;
sigma32 = 11;
sigma33 = 6;
sigma41 = 6;
sigma42 = 11;
sigma43 = 6;
    
%% the main loop of the algorithm

for k = 1:epoch

            disp(['iteration: ' num2str(k)]);

            [OL_T1_C1,OL_T1_C2,OL_T2_C1,OL_T2_C2, Pblock_1, Pblock_2]=sim_AC_v3(lambda_t1_c1(k),lambda_t1_c2(k),lambda_t2_c1(k),lambda_t2_c2(k));
            OL = [OL_T1_C1,OL_T1_C2,OL_T2_C1,OL_T2_C2];
            OL_Cell_1 = [OL(1),OL(3)];
            OL_Cell_2 = [OL(2),OL(4)];
            Pblock = [Pblock_1, Pblock_2];                      

            ST(:,1) = exp(-(OL_Cell_1(1)-mean11)^2/(2*sigma11^2));     % Low OL1
            ST(:,2) = exp(-(OL_Cell_1(1)-mean12)^2/(2*sigma12^2));     % Medium OL1
            ST(:,3) = exp(-(OL_Cell_1(1)-mean13)^2/(2*sigma13^2));     % High OL1

            ST(:,4) = exp(-(OL_Cell_1(2)-mean21)^2/(2*sigma21^2));     % Low OL2
            ST(:,5) = exp(-(OL_Cell_1(2)-mean22)^2/(2*sigma22^2));     % Medium OL2
            ST(:,6) = exp(-(OL_Cell_1(2)-mean23)^2/(2*sigma23^2));     % High OL2

            ST(:,7) = trapmf(delta_T1_C1,trapmf_1);     % Low Delta_T1_C1
            ST(:,8) = trimf(delta_T1_C1,trimf_1);     % Medium Delta_T1_C1
            ST(:,9) = trapmf(delta_T1_C1,trapmf_2);    % High Delta_T1_C1
            
            ST(:,10) = trapmf(delta_T2_C1,trapmf_1);     % Low Delta_T2_C1
            ST(:,11) = trimf(delta_T2_C1,trimf_1);     % Medium Delta_T2_C1
            ST(:,12) = trapmf(delta_T2_C1,trapmf_2);    % High Delta_T2_C1

            %% Start Fuzzy variables %%
            FUZZY(:,1) = [ST(:,1) ST(:,4) ST(:,7) ST(:,10)];  % LLLL
            FUZZY(:,2) = [ST(:,1) ST(:,4) ST(:,7) ST(:,11)];  % LLLM
            FUZZY(:,3) = [ST(:,1) ST(:,4) ST(:,7) ST(:,12)];  % LLLH
           
            FUZZY(:,4) = [ST(:,1) ST(:,4) ST(:,8) ST(:,10)];  % LLML
            FUZZY(:,5) = [ST(:,1) ST(:,4) ST(:,8) ST(:,11)];  % LLMM
            FUZZY(:,6) = [ST(:,1) ST(:,4) ST(:,8) ST(:,12)];  % LLMH
              
            FUZZY(:,7) = [ST(:,1) ST(:,4) ST(:,9) ST(:,10)];   % LLHL
            FUZZY(:,8) = [ST(:,1) ST(:,4) ST(:,9) ST(:,11)];  % LLHM
            FUZZY(:,9) = [ST(:,1) ST(:,4) ST(:,9) ST(:,12)];  % LLH
            
            FUZZY(:,10) = [ST(:,1) ST(:,5) ST(:,7) ST(:,10)];  % LMLL
            FUZZY(:,11) = [ST(:,1) ST(:,5) ST(:,7) ST(:,11)];  % LMLM
            FUZZY(:,12) = [ST(:,1) ST(:,5) ST(:,7) ST(:,12)];  % LMLH
            
            FUZZY(:,13) = [ST(:,1) ST(:,5) ST(:,8) ST(:,10)];  % LMML
            FUZZY(:,14) = [ST(:,1) ST(:,5) ST(:,8) ST(:,11)];  % LMMM
            FUZZY(:,15) = [ST(:,1) ST(:,5) ST(:,8) ST(:,12)];  % LMMH
            
            FUZZY(:,16) = [ST(:,1) ST(:,5) ST(:,9) ST(:,10)];  % LMHL
            FUZZY(:,17) = [ST(:,1) ST(:,5) ST(:,9) ST(:,11)];  % LMHM
            FUZZY(:,18) = [ST(:,1) ST(:,5) ST(:,9) ST(:,12)];  % LMHH
            
            FUZZY(:,19) = [ST(:,1) ST(:,6) ST(:,7) ST(:,10)];  % LHLL
            FUZZY(:,20) = [ST(:,1) ST(:,6) ST(:,7) ST(:,11)];  % LHLM
            FUZZY(:,21) = [ST(:,1) ST(:,6) ST(:,7) ST(:,12)];  % LHLH
                        
            FUZZY(:,22) = [ST(:,1) ST(:,6) ST(:,8) ST(:,10)];  % LHML
            FUZZY(:,23) = [ST(:,1) ST(:,6) ST(:,8) ST(:,11)];  % LHMM
            FUZZY(:,24) = [ST(:,1) ST(:,6) ST(:,8) ST(:,12)];  % LHMH
            
            FUZZY(:,25) = [ST(:,1) ST(:,6) ST(:,9) ST(:,10)];  % LHHL
            FUZZY(:,26) = [ST(:,1) ST(:,6) ST(:,9) ST(:,11)];  % LHHM
            FUZZY(:,27) = [ST(:,1) ST(:,6) ST(:,9) ST(:,12)];  % LHHH
            
            FUZZY(:,28) = [ST(:,2) ST(:,4) ST(:,7) ST(:,10)];  % MLLL
            FUZZY(:,29) = [ST(:,2) ST(:,4) ST(:,7) ST(:,11)];  % MLLM
            FUZZY(:,30) = [ST(:,2) ST(:,4) ST(:,7) ST(:,12)];  % MLLH
            
            FUZZY(:,31) = [ST(:,2) ST(:,4) ST(:,8) ST(:,10)];  % MLML
            FUZZY(:,32) = [ST(:,2) ST(:,4) ST(:,8) ST(:,11)];  % MLMM
            FUZZY(:,33) = [ST(:,2) ST(:,4) ST(:,8) ST(:,12)];  % MLMH
            
            FUZZY(:,34) = [ST(:,2) ST(:,4) ST(:,9) ST(:,10)];  % MLHL
            FUZZY(:,35) = [ST(:,2) ST(:,4) ST(:,9) ST(:,11)];  % MLHM
            FUZZY(:,36) = [ST(:,2) ST(:,4) ST(:,9) ST(:,12)];  % MLHH
                        
            FUZZY(:,37) = [ST(:,2) ST(:,5) ST(:,7) ST(:,10)];  % MMLL
            FUZZY(:,38) = [ST(:,2) ST(:,5) ST(:,7) ST(:,11)];  % MMLM
            FUZZY(:,39) = [ST(:,2) ST(:,5) ST(:,7) ST(:,12)];  % MMLH
            
            FUZZY(:,40) = [ST(:,2) ST(:,5) ST(:,8) ST(:,10)];  % MMML
            FUZZY(:,41) = [ST(:,2) ST(:,5) ST(:,8) ST(:,11)];  % MMMM
            FUZZY(:,42) = [ST(:,2) ST(:,5) ST(:,8) ST(:,12)];  % MMMH
                        
            FUZZY(:,43) = [ST(:,2) ST(:,5) ST(:,9) ST(:,10)];  % MMHL
            FUZZY(:,44) = [ST(:,2) ST(:,5) ST(:,9) ST(:,11)];  % MMHM
            FUZZY(:,45) = [ST(:,2) ST(:,5) ST(:,9) ST(:,12)];  % MMHH
                  
            FUZZY(:,46) = [ST(:,2) ST(:,6) ST(:,7) ST(:,10)];  % MHLL
            FUZZY(:,47) = [ST(:,2) ST(:,6) ST(:,7) ST(:,11)];  % MHLM
            FUZZY(:,48) = [ST(:,2) ST(:,6) ST(:,7) ST(:,12)];  % MHLH
            
            FUZZY(:,49) = [ST(:,2) ST(:,6) ST(:,8) ST(:,10)];  % MHML
            FUZZY(:,50) = [ST(:,2) ST(:,6) ST(:,8) ST(:,11)];  % MHMM
            FUZZY(:,51) = [ST(:,2) ST(:,6) ST(:,8) ST(:,12)];  % MHMH
            
            FUZZY(:,52) = [ST(:,2) ST(:,6) ST(:,9) ST(:,10)];  % MHHL
            FUZZY(:,53) = [ST(:,2) ST(:,6) ST(:,9) ST(:,11)];  % MHHM
            FUZZY(:,54) = [ST(:,2) ST(:,6) ST(:,9) ST(:,12)];  % MHHH
            
            FUZZY(:,55) = [ST(:,3) ST(:,4) ST(:,7) ST(:,10)];   % HLLL
            FUZZY(:,56) = [ST(:,3) ST(:,4) ST(:,7) ST(:,11)];   % HLLM
            FUZZY(:,57) = [ST(:,3) ST(:,4) ST(:,7) ST(:,12)];   % HLLH
            
            FUZZY(:,58) = [ST(:,3) ST(:,4) ST(:,8) ST(:,10)];  % HLML
            FUZZY(:,59) = [ST(:,3) ST(:,4) ST(:,8) ST(:,11)];  % HLMM
            FUZZY(:,60) = [ST(:,3) ST(:,4) ST(:,8) ST(:,12)];  % HLMH
            
            FUZZY(:,61) = [ST(:,3) ST(:,4) ST(:,9) ST(:,10)];  % HLHL
            FUZZY(:,62) = [ST(:,3) ST(:,4) ST(:,9) ST(:,11)];  % HLHM
            FUZZY(:,63) = [ST(:,3) ST(:,4) ST(:,9) ST(:,12)];  % HLHH
            
            FUZZY(:,64) = [ST(:,3) ST(:,5) ST(:,7) ST(:,10)];  % HMLL
            FUZZY(:,65) = [ST(:,3) ST(:,5) ST(:,7) ST(:,11)];  % HMLM
            FUZZY(:,66) = [ST(:,3) ST(:,5) ST(:,7) ST(:,12)];  % HMLH
            
            FUZZY(:,67) = [ST(:,3) ST(:,5) ST(:,8) ST(:,10)];  % HMML
            FUZZY(:,68) = [ST(:,3) ST(:,5) ST(:,8) ST(:,11)];  % HMMM
            FUZZY(:,69) = [ST(:,3) ST(:,5) ST(:,8) ST(:,12)];  % HMMH
            
            FUZZY(:,70) = [ST(:,3) ST(:,5) ST(:,9) ST(:,10)];  % HMHL
            FUZZY(:,71) = [ST(:,3) ST(:,5) ST(:,9) ST(:,11)];  % HMHM
            FUZZY(:,72) = [ST(:,3) ST(:,5) ST(:,9) ST(:,12)];  % HMHH
            
            FUZZY(:,73) = [ST(:,3) ST(:,6) ST(:,7) ST(:,10)];  % HHLL
            FUZZY(:,74) = [ST(:,3) ST(:,6) ST(:,7) ST(:,11)];  % HHLM
            FUZZY(:,75) = [ST(:,3) ST(:,6) ST(:,7) ST(:,12)];  % HHLH
            
            FUZZY(:,76) = [ST(:,3) ST(:,6) ST(:,8) ST(:,10)];  % HHML
            FUZZY(:,77) = [ST(:,3) ST(:,6) ST(:,8) ST(:,11)];  % HHMM
            FUZZY(:,78) = [ST(:,3) ST(:,6) ST(:,8) ST(:,12)];  % HHMH
                        
            FUZZY(:,79) = [ST(:,3) ST(:,6) ST(:,9) ST(:,10)];  % HHHL
            FUZZY(:,80) = [ST(:,3) ST(:,6) ST(:,9) ST(:,11)];  % HHHM
            FUZZY(:,81) = [ST(:,3) ST(:,6) ST(:,9) ST(:,12)];  % HHHH
            
            %% End Fuzzy variables %%

            for i=1:81
                alpha_i(k,i) = FUZZY(1,i)*FUZZY(2,i)*FUZZY(3,i)*FUZZY(4,i);
            end

            [strength_max,state_idx_max] = max(alpha_i(k,:));

            if k~=1
                state_idx(k) = state_idx_max;
                % Compute the value of the new state %
                for i=1:length(state)
                        V_t(state_idx(k)) = alpha_i(k,i)*max(q1(i,action_idx(k-1))) + V_t(state_idx(k));
                end
                % Calculate the error signal %
                Q_error1(k) = reward(k) + gamma*V_t(state_idx(k))-Q1(state_idx(k-1),action_idx(k-1));

                % Update q-values by an ordinary gradient descent method %
                q1(state_idx(k),umax(i)) = q1(state_idx(k),umax(i)) + eta*Q_error1(k)*alpha_i(k-1,state_idx(k));
            end    

            r=rand; % get 1 uniform random number
            prob_area=sum(r>=cumsum([0, 1-epsilon, epsilon])); % check it to be in which probability area

            for i=1:length(state)
                % choose either explore or exploit
                if prob_area == 1   % exploit
                        [~,umax(i)]=max(q1(i,:));
                        a_i(i) = action(umax(i));
                else        % explore
                    [a_i(i),umax(i)] = datasample(action,1); % choose 1 action randomly (uniform random distribution)
                end
            end
            
            % Calculate the global action %

            for i=1:length(state)
                act(k) = act(k) + alpha_i(k,i)*a_i(i);
            end

            [c index] = min(abs(action-act(k)));
            action_idx(k) = find(action==action(index)); % id of the chosen action

            % Approximate the Q-function from the current q-values and the degree
            % of truth of the rules 
            
            for i=1:length(state)
                        Q1(state_idx(k),action_idx(k)) = alpha_i(k,i)*q1(i,umax(i))+Q1(state_idx_max,action_idx(k));
            end
            
            for(i=1:81)
                q_values1(i,k) = Q1(i,1);
                q_values2(i,k) = Q1(i,2);
                q_values3(i,k) = Q1(i,3);
                q_values4(i,k) = Q1(i,4);
                q_values5(i,k) = Q1(i,5);
            end
                                    
            %% Evolve to the next state %%

            % Observe the reinforcement signal %

            delta_T1_C1 = delta_T1_C1 + action(action_idx(k));
            
            delta_T1_C2 = evalfis(OL,fismat_T1_C2);
            delta_T2_C1 = evalfis(OL,fismat_T2_C1);
            delta_T2_C2 = evalfis(OL,fismat_T2_C2);

            [Pblock1(k+1),Pblock2(k+1),Pblock11(k+1),Pblock12(k+1),Pblock21(k+1),Pblock22(k+1)]=sim_AC_v4(lambda_t1_c1(k),lambda_t1_c2(k),lambda_t2_c1(k),lambda_t2_c2(k),delta_T1_C1,delta_T1_C2,delta_T2_C1,delta_T2_C2);
            [Pblock1_nd(k+1),Pblock2_nd(k+1),Pblock11_nd(k+1),Pblock12_nd(k+1),Pblock21_nd(k+1),Pblock22_nd(k+1)]=sim_AC_v6(lambda_t1_c1(k),lambda_t1_c2(k),lambda_t2_c1(k),lambda_t2_c2(k),delta_T1_C1,delta_T1_C2,delta_T2_C1,delta_T2_C2);

            r1(k) = log10((1+1/((Pblock1(k+1)+0.1)*1000)));
            r2(k) = log10((1+1/((Pblock2(k+1)+0.1)*1000)));

            reward(k+1) = 100*(r1(k)+r2(k))+0.1357;
           
            % conditions
            if (delta_T1_C1 <= -0.3 || delta_T1_C1 >= 0.65)
                delta_T1_C1 = 0.2; 
            end
                                            
          % Update epsilon
          epsilon = epsilon - (1/650);
end
