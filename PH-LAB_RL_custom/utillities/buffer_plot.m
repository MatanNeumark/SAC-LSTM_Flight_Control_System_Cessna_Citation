buffer = SAC_agent.buffer;

%% reward
name_vec = [" - \beta", " - \phi", " - \theta"];
figure
tiledlayout;

nexttile
hold on
grid on
plot(buffer.reward, LineWidth=1, color='m')
legend( 'reward')
ylim([-0.5 0.5])
for i = 1:length(SAC_agent.env.tracked_idx)
nexttile
hold on
grid on
plot(buffer.state(:,SAC_agent.env.tracked_idx(i)) - buffer.state(:,length(SAC_agent.env.state_vars)+i))
legend(SAC_agent.env.tracked_states_dict(SAC_agent.env.tracked_states(i)) + name_vec(i))
ylim([-0.3 0.3])

end

% p_nMAE = mean(abs(state(:,8)- state(:,1))) / mean(abs(state(:,8)))
% q_nMAE = mean(abs(state(:,9)- state(:,2))) / mean(abs(state(:,9)))
% beta_nMAE = mean(abs(state(:,10)- state(:,5))) / mean(abs(state(:,10)))
% 
% mean_e_p = mean(abs(buffer.state(:,8) - buffer.state(:,5)))
% mean_e_q = mean(abs(buffer.state(:,9) - buffer.state(:,6)))
% mean_e_beta = mean(abs(buffer.state(:,10) - buffer.state(:,7)))

%% elevator
figure
hold on
grid on
plot(buffer.state(:,11))
yline(0)
legend('\delta_e')
%% aileron
figure
hold on
grid on
plot(buffer.state(:,12))
yline(0)
legend('\delta_a')
%% rudder
figure
hold on
grid on
plot(buffer.state(:,13))
yline(0)
legend('\delta_r')
%% p
figure
hold on
grid on
plot(buffer.state(:,1))
%plot(buffer.state(:,7))
yline(0)
mean_p = mean(abs(buffer.state(:,1)))
%% q
figure
hold on
grid on
plot(buffer.state(:,2))
%plot(buffer.state(:,9))
yline(0)
%% r
figure
hold on
grid on
plot(buffer.state(:,3))
yline(0)
%% beta
figure
hold on
grid on
plot(buffer.state(:,5))
plot(buffer.state(:,8))
legend('\beta', '\beta_{ref}')
mean_beta = mean(abs(buffer.state(:,5)))

%% phi
figure
hold on
grid on
plot(buffer.state(:,6))
plot(buffer.state(:,9))
legend('\phi', '\phi_{ref}')
mean_phi = mean(abs(buffer.state(:,6)))

%% theta
figure
hold on
grid on
plot(buffer.state(:,7))
plot(buffer.state(:,10))
legend('\theta', '\theta_{ref}')
mean_theta = mean(abs(buffer.state(:,7)))
%%
p_exceeded = nnz(buffer.new_state(:,1)>1)
q_exceeded = nnz(buffer.new_state(:,2)>1)
r_exceeded = nnz(buffer.new_state(:,3)>1)
alpha_exceeded = nnz(buffer.new_state(:,4)>1)
beta_exceeded = nnz(buffer.new_state(:,5)>1)
phi_exceeded = nnz(buffer.new_state(:,6)>1)
theta_exceeded = nnz(buffer.new_state(:,7)>1)

%% action
figure
hold on
grid on
plot(buffer.action(:,1))
%plot(buffer.action(:,2))
%plot(buffer.action(:,3))
yline(0)
%% all states
figure
hold on
grid on
for i =1:SAC_agent.bounds.state_space_size-1
    plot(buffer.state(:,i))
end
legend([SAC_agent.env.state_vars_dict(SAC_agent.env.state_vars), SAC_agent.env.tracked_states_dict(SAC_agent.env.tracked_states)])
