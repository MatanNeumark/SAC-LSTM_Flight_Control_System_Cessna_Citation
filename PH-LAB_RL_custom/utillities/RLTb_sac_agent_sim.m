% clc
% clear
% load('SAC_data.mat')
Env_config
env_mdl = "Citation_RL_custom_env_by_Matan"; 
steps_per_episode = 1000;
dt = 0.01;
episode_length = steps_per_episode * dt;
time = (0:dt:episode_length);
%state_vars = [1, 2, 3, 6, 7, 8]; % p, q, r, alpha, beta, phi, theta
state_vars = [5, 2]; % p, q, r, alpha, beta, phi, theta
%tracked_states = [1, 2, 6]; % p, q, beta
tracked_states = 2; % p, q, beta

%q_ref = deg2rad(5)*sin(2*pi*0.2*time);
%beta_ref = deg2rad(1)*sin(0.3*pi*time);
beta_ref = zeros(1,length(time));
%phi_ref = deg2rad(2)*sin(0.2*pi*time);
p_ref = zeros(1,length(time));
q_ref = deg2rad(5)*sin(0.4*pi*time);
%q = zeros(1,length(time));
ref_state = q_ref';
% ref_state = [p_ref', q_ref', beta_ref'];
%q_ref = zeros(1, length(time));
RF = [3, 4, 5, 2, 1];
% env_settings = containers.Map({'ref_state', 'dt', 'longitudinal', 'linear', 'steps_per_episode', 'RF'}, ...
%                               {q_ref,        dt,   true,           true,     steps_per_episode,   RF(run)});
env_settings = containers.Map({'ref_state', 'dt', 'steps_per_episode', 'RF',  'state_vars', 'tracked_states'}, ...
                               {ref_state,   dt,   steps_per_episode,   RF(4), state_vars,   tracked_states});

env = Linear_Citation_env(env_settings);
%alt_env = Linear_Citation_env_alt_dynamics(env_settings)
%env = Citation_env(env_mdl, env_settings);

sac = SAC(env);

% RL_controller = RLTb_reward6_seed_91.actorNet;
% agent = RLTb_reward6_seed_91.agent;
agent.UseExplorationPolicy = 0;

%%
simOpts = rlSimulationOptions(MaxSteps=1000);
experience = sim(sac.env,agent,simOpts);

% sim_alpha = experience.Observation.CitationStates.Data(1,1,:);
% sim_q = experience.Observation.CitationStates.Data(1,2,:);
% sim_q_ref = experience.Observation.CitationStates.Data(1,3,:);
% sim_action = experience.Action.Elevator.Data(1,1,:);
% sim_delta_action = sim_action(2:end) - sim_action(1:end-1);
% sim_nMAE = mean(abs(sim_q_ref - sim_q)) / mean(abs(sim_q_ref));
% sim_MTV = mean(abs(sim_delta_action));
%%
% RLTb_reward6_seed_91_3211.sim_alpha = sim_alpha;
% RLTb_reward6_seed_91_3211.sim_q = sim_q;
% RLTb_reward6_seed_91_3211.sim_q_ref = sim_q_ref;
% RLTb_reward6_seed_91_3211.sim_action = sim_action;
% RLTb_reward6_seed_91_3211.sim_nMAE = sim_nMAE;
% RLTb_reward6_seed_91_3211.sim_MTV = sim_MTV;
%%
experience.Observation.CitationStates.Data = squeeze(experience.Observation.CitationStates.Data);
figure
hold on
for i = 1:obsInfo.Dimension(2)
    plot(experience.Observation.CitationStates.Data(i,:), LineWidth=1)
end

%xlim([0,length(time)])
legend('p', 'q', 'r', '\beta', '\phi', '\theta', '\beta_{ref}', '\phi_{ref}', '\theta_{ref}', 'e_{\beta}', 'e_{\phi}', 'e_{\theta}')
ylabel('[rad]')
xlabel('time step')
%title("sim nMAE(q,q_{ref}) = " + sim_nMAE)
%ylim([-0.2 + (min(sim_q_ref)), 0.2 + (max(sim_q_ref))])
grid on


%%
sim_delta_action = double(gather(sim_action(2:end) - sim_action(1:end-1)));
sim_nMAE = mean(abs(sim_q- sim_q_ref)) / mean(abs(sim_q_ref))
sim_MTV =  mean(abs(delta_action))

