clc
clear
state_vars_dict = dictionary(1, "p", 2, "q", 3, "r", 4, "V_TAS", 5, "\alpha", 6, "\beta", 7, "\phi", 8, "\theta", 9,  "\psi", 10, "he", 11, "xe", 12, "ye");
tracked_states_dict = dictionary(1, "p_{ref}", 2, "q_{ref}", 3, "r_{ref}", 4, "V_TAS_{ref}", 5, "\alpha_{ref}", 6, "\beta_{ref}", 7, "\phi_{ref}", 8, "\theta_{ref}", 9,  "\psi_{ref}", 10, "he_{ref}", 11, "xe_{ref}", 12, "ye_{ref}");

steps_per_episode = 1000;
dt = 0.01;
episode_length = steps_per_episode * dt;
time = 0:dt:episode_length;
%state_vars = [1, 2, 3, 5, 6, 7, 8]; % p, q, r, alpha, beta, phi, theta
state_vars = [2, 5, 8]; % q, alpha, theta
tracked_states = 8; % theta
priming = 200;
%ref_state = [zeros(priming, 1); deg2rad(5)*sin(0.2*pi*time(1:end-priming))'];
%ref_state = deg2rad(5)*sin(0.2*pi*time)';
ref_state = [zeros(priming, 1); deg2rad(5)*ones(steps_per_episode - priming + 1,1)];

SA_noise_bias   = false;
SA_delay        = true;
servo_RL        = false;
servo_TF        = true;
training        = false;
incremental     = false;
env = Linear_Citation_env_GRU(ref_state, dt, steps_per_episode, state_vars, tracked_states, SA_noise_bias, SA_delay, incremental, training, servo_RL, servo_TF);
%load ("C_outer.mat")
%load ("C_inner.mat")
%controller = PID_controller(dt, [], []);
%[C_outer, C_inner] = controller.auto_tune(env.lin_sys, [], 10, 10, [], 100, false)


% Best gains to date with nMAE<5% in a clean environment.
% It Can deal with noise, bias, and delay.
C_outer.Kp = 8;
C_outer.Ki = 12;
C_outer.Kd = 0.2;
C_outer.N = 100;
C_inner.Kp = 1.8;
C_inner.Ki = 0;
C_inner.Kd = 0.05;%0.1;
C_inner.N = 25;


% C_outer.Kp = 1.5;
% C_outer.Ki = 0;
% C_outer.Kd = 0;
% C_inner.Kp = 0.4;
% C_inner.Ki = 1;
% C_inner.Kd = 0;
%%
%load("C_inner.mat")
%load("C_outer.mat")
controller = PID_controller(dt, steps_per_episode, C_outer, C_inner);



seed = 6;
%rng(seed, "twister")
state = [];
action = [];
reward = [];
applied_action = [];
internal_state = [];
[state(1,:), action(1,:)] = env.reset;
controller.reset
%applied_action(1,:) = action(1,:);

%internal_state(1,:) = state(1,1:2);
for t = 1:steps_per_episode
    t
    action(t+1,:) = controller.control(state(t, 1), state(t, 3), state(t, 4));
    action(t+1,:) = clip(action(t+1,:), env.lo_sat_lim, env.up_sat_lim);
    [state(t+1,:), reward(t), ~, ~, applied_action(t+1), internal_state(t+1,:)] = env.step(action(t+1,:));
end
applied_action = applied_action';
env.stop
controller.stop



%delta_action = double(gather(action(2:end) - action(1:end-1)));
%nMAE = mean(abs(state(:,3)- state(:,1))) / mean(abs(state(:,3)))
nMAE = mean(abs(state(:,4)- state(:,3))) / mean(abs(state(:,4)))
%theta_nMAE = mean(abs(state(:,10)- state(:,7))) / mean(abs(state(:,10)))
%phi_nMAE = mean(abs(state(:,9)- state(:,6))) / mean(abs(state(:,9)))
%beta_nMAE = mean(abs(state(:,8)- state(:,5))) / mean(abs(state(:,8)))

%MTV =  mean(abs(delta_action));


 %all states
color = ['k', 'b', 'm', 'g', 'r', 'c', "#4DBEEE", "#77AC30", "#7E2F8E", "#EDB120", "#D95319", "#D95310", "#D95610"];
linestyle = ["--", ":", "-."];
figure
hold on
grid on
for i = 1:width(state)
    plot((state(:,i)), LineWidth=1, Color=color(i))
end
for i = 1:3%width(internal_state)
    plot((internal_state(:,i)), LineWidth=1, Color=color(i), LineStyle=":")
end

% plot(ref_state(:,1), Color=color(8))
% plot(ref_state(:,2), Color=color(9))
% plot(ref_state(:,3), Color=color(10))
for i = 1:width(action)
    plot(action(:,i), LineWidth=0.1, Color=[0.1 0.1 0.1])
    plot(applied_action(:,i), LineWidth=0.1, Color=[0.1 0.1 0.1], LineStyle=":")
end
legend([state_vars_dict(env.state_vars), tracked_states_dict(env.tracked_states), "", "", "", "\delta_e", "applied \delta_e"])
%leg1 = legend('$\hat{q}$', '$\hat{\alpha}$', '$q_{ref}$', '$q$', '$\alpha$', '$\delta_e$');
%set(leg1,'Interpreter','latex');
%set(leg1,'FontSize',12);
% title("PID controller in linear env, nMAE(\theta,\theta_{ref}) = " + 100*nMAE + "%")
title("PID, lin env, nMAE(\theta,\theta_{ref}) = " + 100*nMAE + "%,"...
    + " Kp_o=" + num2str(C_outer.Kp) ...
    + " Ki_o=" + num2str(C_outer.Ki) ...
    + " Kd_o=" + num2str(C_outer.Kd) ...
    + " Kp_i=" + num2str(C_inner.Kp) ...
    + " Ki_i=" + num2str(C_inner.Ki) ...
    + " Kd_i=" + num2str(C_inner.Kd))

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
