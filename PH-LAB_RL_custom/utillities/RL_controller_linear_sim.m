% clc
% clear
clearvars -except SAC_agents SAC_agent stats
%%
%SAC_agent = SAC_agents{3};

%%
SAC_agent = FF_SAC_agents{1,10}; idx=37;
%SAC_agent = LSTM_SAC_agents{1,6}; idx=32;
steps_per_episode = 1000;
dt = 0.01;
episode_length = steps_per_episode * dt;
time = 0:dt:episode_length;
priming = 200;
%ref_state = [zeros(priming, 1); deg2rad(5)*sin(0.2*pi*time(1:end-priming))'];
%ref_state = deg2rad(5)*sin(0.2*pi*time)';
ref_state = [zeros(priming, 1); deg2rad(5)*ones(steps_per_episode - priming + 1,1)];


% state_vars_dict = dictionary(1, "p", 2, "q", 3, "r", 4, "V_TAS", 5, "\alpha", 6, "\beta", 7, "\phi", 8, "\theta", 9,  "\psi", 10, "he", 11, "xe", 12, "ye");
% tracked_states_dict = dictionary(1, "p_{ref}", 2, "q_{ref}", 3, "r_{ref}", 4, "V_TAS_{ref}", 5, "\alpha_{ref}", 6, "\beta_{ref}", 7, "\phi_{ref}", 8, "\theta_{ref}", 9,  "\psi_{ref}", 10, "he_{ref}", 11, "xe_{ref}", 12, "ye_{ref}");
state_vars_dict = dictionary(1, "$p$", 2, "$q$", 3, "$r$", 4, "$V_{TAS}$", 5, "$\alpha$", 6, "$\beta$", 7, "$\phi$", 8, "$\theta$", 9,  "$\psi$", 10, "he", 11, "xe", 12, "ye",...
                             13, "$\dot{p}$", 14, "$\dot{q}$", 15, "$\dot{r}$", 16, "$\dot{V}_{TAS}$", 17, "$\dot{\alpha}$", 18, "$\dot{\beta}$", 19, "$\dot{\phi}$", 20, "$\dot{\theta}$", 21,  "$\dot{\psi}$", 22, "\dot{he}", 23, "\dot{xe}", 24, "\dot{ye}");
po_state_vars_dict = dictionary(1, "$\hat{p}$", 2, "$\hat{q}$", 3, "$\hat{r}$", 4, "$\hat{V}_TAS$", 5, "$\hat{\alpha}$", 6, "$\hat{\beta}$", 7, "$\hat{\phi}$", 8, "$\hat{\theta}$", 9,  "$\hat{\psi}$", 10, "$\hat{he}$", 11, "$\hat{xe}$", 12, "$\hat{ye}$");
tracked_states_dict = dictionary(1, "$p_{ref}$", 2, "$q_{ref}$", 3, "$r_{ref}$", 4, "$V_{TAS,ref}$", 5, "$\alpha_{ref}$", 6, "$\beta_{ref}$", 7, "$\phi_{ref}$", 8, "$\theta_{ref}$", 9,  "$\psi_{ref}$", 10, "$he_{ref}$", 11, "$xe_{ref}$", 12, "$ye_{ref}$");
actuator_dict = dictionary(1, "$\delta_e$", 2, "$\delta_a$", 3, "$\delta_r$");
po_actuator_dict = dictionary(1, "$\hat{\delta}_e$", 2, "$\hat{\delta}_a$", 3, "$\hat{\delta}_r$");

% [max_return,idx] = max(mean(SAC_agent.datalogger.ep_return ./ SAC_agent.datalogger.ep_length, 2))
% [max_return,idx] = max(SAC_agent.datalogger.ep_return ./ SAC_agent.datalogger.ep_length, [], "all");
% [row, col] = ind2sub(size(SAC_agent.datalogger.ep_return), idx);
%idx = 37
%idx = stats{1,5}.idx;
SAC_agent.env.steps_per_episode = steps_per_episode;
deterministic = true;
%SAC_agent.env = env;
SAC_agent.env.ref_state = ref_state;
SAC_agent.env.training = false;

SAC_agent.env.SA_noise_bias = false;
SAC_agent.env.SA_delay = true;
SAC_agent.env.servo_RL = false;
SAC_agent.env.servo_TF = true;
seed = 6;
%rng(seed, "twister")
%state(1,:) = SAC_agent.env.reset;
%for ep = 1:200
%ep
state = [];
action = [];
reward = [];
applied_action = [];
internal_state = [];
[state(1,:), action(1,:)] = SAC_agent.env.reset;
applied_action(1,:) = action(1,:);

SAC_agent.datalogger.actor.policy.net(idx) = resetState(SAC_agent.datalogger.actor.policy.net(idx));
%internal_state(1,:) = state(1,1:2);
for t = 1:steps_per_episode
    t
    %action(t+1,:) = SAC_agent.predict(state(t,:), deterministic, SAC_agent.datalogger.actor.policy.net(idx));
    [action(t+1,:), hs] = SAC_agent.predict(state(t,:), action(t,:), deterministic, SAC_agent.datalogger.actor.policy.net(idx));
    SAC_agent.datalogger.actor.policy.net(idx).State = hs;
    [state(t+1,:), reward(t), ~, ~, applied_action(t+1,:), internal_state(t+1,:)] = SAC_agent.env.step(action(t+1,:));
end
%applied_action = applied_action';
SAC_agent.env.stop

%LSTM_TC2_stats.CA(ep) = trapz(dt, abs(diff(action)/dt)) / episode_length
CA = trapz(dt, abs(diff(action)/dt)) / episode_length;
ep_return = sum(reward);

%nMAE = mean(abs(state(:,3)- state(:,1))) / mean(abs(state(:,3)))
%LSTM_TC2_stats.nMAE(ep) = mean(abs(state(:,4)- state(:,3))) / mean(abs(state(:,4)))
nMAE = mean(abs(state(:,4)- state(:,3))) / mean(abs(state(:,4)));
%theta_nMAE = mean(abs(state(:,10)- state(:,7))) / mean(abs(state(:,10)))
%phi_nMAE = mean(abs(state(:,9)- state(:,6))) / mean(abs(state(:,9)))
%beta_nMAE = mean(abs(state(:,8)- state(:,5))) / mean(abs(state(:,8)))

%alt_ctrl_activity =  sum(abs(diff(action))/dt)/steps_per_episode
%end



% all states
color = ['k', 'b', 'm', 'g', 'r', 'c', "#4DBEEE", "#77AC30", "#7E2F8E", "#EDB120", "#D95319", "#D95310", "#D95610"];
linestyle = ["--", ":", "-."];
figure
hold on
grid on
for i = 1:4%width(state)
    plot((state(:,i)), LineWidth=0.1, Color=color(i))
end
for i = 1:width(internal_state)
    p = plot((internal_state(:,i)), LineWidth=0.1, Color=color(i));
    p.Color(4) = 0.2;
end

% plot(ref_state(:,1), Color=color(8))
% plot(ref_state(:,2), Color=color(9))
% plot(ref_state(:,3), Color=color(10))
for i = 1:width(action)
    plot(action(:,i), LineWidth=0.1, Color=[0 0.8 0.8])
    plot(applied_action(:,i), LineWidth=0.1, Color=[0 0.8 0.8, 0.5])
end
if SAC_agent.env.SA_delay == true
    leg1 = legend([po_state_vars_dict(SAC_agent.env.state_vars), tracked_states_dict(SAC_agent.env.tracked_states), state_vars_dict(SAC_agent.env.state_vars), "$\delta_e$", "applied $\delta_e$"]);
else
    leg1 = legend([state_vars_dict(SAC_agent.env.state_vars), tracked_states_dict(SAC_agent.env.tracked_states), "", "", "", "$\delta_e$", "applied $\delta_e$"]);
end
set(leg1,'Interpreter','latex');
%set(leg1,'FontSize',12);
title("nMAE(\theta,\theta_{ref}) = " + 100*nMAE + "%, control activity = " + CA + " idx=" + num2str(idx))

xlim([0,steps_per_episode+1])
ylabel('[rad], [rad/s]')
xlabel('time step')
%title("nMAE(q,q_{ref}) = " + 100*nMAE + "%")
ylim([-0.1 0.1])

%% subplots for lateral and longitudinal motions
theta_nMAE = mean(abs(state(:,10)- state(:,7))) / mean(abs(state(:,10)))
phi_nMAE = mean(abs(state(:,9)- state(:,6))) / mean(abs(state(:,9)))
beta_nMAE = mean(abs(state(:,8)- state(:,5))) / mean(abs(state(:,8)))
phi_MAE = mean(abs(state(:,9)- state(:,6)))
beta_MAE = mean(abs(state(:,8)- state(:,5)))
color = ['k', 'b', 'm', 'g', 'r', 'c', "#4DBEEE", "#77AC30", "#7E2F8E", "#EDB120", "#D95319", "#D95310"];
linestyle = ["--", ":", "-."];
figure
tiledlayout(3,2, "TileSpacing","tight")
nexttile
title("nMAE(\theta,\theta_{ref}) = " + 100*theta_nMAE + "%")
hold on
grid on
plot((state(:,2)), LineWidth=1, Color=color(2))
plot((state(:,4)), LineWidth=1, Color=color(5))
plot((state(:,7)), LineWidth=1, Color=color(7))
plot((state(:,10)), LineWidth=1, Color=color(10))
legend([state_vars_dict([2, 5, 8]), tracked_states_dict(8)])
xlim([0,steps_per_episode+1])
ylim([-0.15 0.15])
xlabel('time step')
ylabel('[rad], [rad/s]')
nexttile
title("\delta_e")
hold on
grid on
plot(action(:,1), LineWidth=1, Color='k', LineStyle=linestyle(1))
legend("\delta_e")
xlim([0,steps_per_episode+1])
ylim([-0.15 0.15])
xlabel('time step')
ylabel('[rad], [rad/s]')
nexttile
title("nMAE(\phi,\phi_{ref}) = " + 100*phi_nMAE + "%")
hold on
grid on
plot((state(:,1)), LineWidth=1, Color=color(1))
plot((state(:,6)), LineWidth=1, Color=color(6))
plot((state(:,9)), LineWidth=1, Color=color(9))
legend([state_vars_dict([1, 7]), tracked_states_dict(7)])
xlim([0,steps_per_episode+1])
ylim([-0.2 0.2])
xlabel('time step')
ylabel('[rad], [rad/s]')
nexttile
title("\delta_a")
hold on
grid on
plot(action(:,2), LineWidth=1, Color='k', LineStyle=linestyle(1))
legend("\delta_a")
xlim([0,steps_per_episode+1])
ylim([-0.2 0.2])
xlabel('time step')
ylabel('[rad], [rad/s]')
nexttile
title("MAE(\beta,\beta_{ref}) = " + beta_MAE)
hold on
grid on
plot((state(:,3)), LineWidth=1, Color=color(3))
plot((state(:,5)), LineWidth=1, Color=color(5))
plot((state(:,8)), LineWidth=1, Color=color(8))
legend([state_vars_dict([3, 6]), tracked_states_dict(6)])
xlim([0,steps_per_episode+1])
ylim([-0.05 0.05])
xlabel('time step')
ylabel('[rad], [rad/s]')
nexttile
title("\delta_r")
hold on
grid on
plot(action(:,3), LineWidth=1, Color='k', LineStyle=linestyle(1))
legend("\delta_r")
xlim([0,steps_per_episode+1])
ylim([-0.05 0.05])
xlabel('time step')
ylabel('[rad], [rad/s]')
% for i = 1:width(action)
%     plot(action(:,i), LineWidth=1, Color='k', LineStyle=linestyle(i))
% end

% xlim([0,steps_per_episode+1])
% ylabel('[rad], [rad/s], [m]')
% xlabel('time step')
% %title("nMAE(q,q_{ref}) = " + 100*nMAE + "%")
% grid on
% ylim([-0.1 0.1])
