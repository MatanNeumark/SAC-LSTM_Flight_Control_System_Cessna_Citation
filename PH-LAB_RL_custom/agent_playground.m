%% This script instantiates the environment and SAC framework and runs the training.
%% For the environment, choose the degrees of motion (longitudinal, or longitudinal and lateral) by selecting the relevant state variables.
%% Additionally, set the desired state variables to track, and the sensor and actuator dynamics to enable.
%% The hyperparameters and network type (LSTM or feedforward) can be set below, after the environment configuration.

clear
clc
myCluster = parcluster('Processes');
delete(myCluster.Jobs)
%%
n_runs = 1; % number of agents to train (not in parallel)
seed_vec = randperm(1000,n_runs); % chooses random seeds for reproducibility
%%
for run = 1:n_runs
    
    clearvars -except run seed_vec n_runs
    close all
    clc

    env_mdl = "Citation_RL_custom_env"; % needed when using the Simulink environment
    %seed = 956;
    seed = seed_vec(run);
    rng(seed, "twister")
    steps_per_episode = 1000; % number of time steps per episode
    dt = 0.01; % time length of step, in seconds
    episode_length = steps_per_episode * dt; % episode length in seconds
    stop_condition = 0; % mean reward value to stop training. value=0 means no stop condition
    time = 0:dt:episode_length;
    %state_vars = [1, 2, 3, 5, 6, 7, 8]; % p, q, r, alpha, beta, phi, theta. state variables for full DoF simulation 
    state_vars = [2, 5, 8]; % q, alpha, theta. state variables for longitudinal simulation 
    %state_vars = [2, 5]; % q, alpha. for pitch rate control, so the theta is excluded
    %tracked_states = 2; % q. pitch rate control
    tracked_states = 8; % theta. pitch angle control
    %tracked_states = [6, 7, 8]; % beta, phi, theta. for full DoF angle control
    %beta_ref = zeros(1,length(time)); % reference signal
    %phi_ref = zeros(1,length(time)); % reference signal
    %q_ref = deg2rad(5)*sin(0.4*pi*time); % reference signal
    %q_ref = ones(1,length(time)); % reference signal
    %ref_state = q_ref';
    theta_ref = deg2rad(5)*sin(0.4*pi*time); % reference signal
    ref_state = theta_ref'; % reference signal
    %ref_state = [beta_ref', phi_ref', theta_ref']; % reference signal

    %% Choose which sensor and actuator dynamics to enable in the environment
    SA_noise_bias   = false; % sensors and servos noise and bias. 'False' is recommended for training
    SA_delay        = false; % sensors and servos delay. 'False' is recommended for training
    incremental     = false; % incremental control
    servo_RL        = false; % servo rate limit. 'False' is recommended for training because it hinders exploration
    servo_TF        = true; % First order transfer function to model the servo. 'True' is recommended for training so the agent can learn these dynamics
    training        = true; % if 'true', the reference signal frequency and amplitude is chosen at random every episode, otherwise it uses the supplied ref_state. "true" also enables exploring starts.

    %% Instantiate the environment
    env = Linear_Citation_env_for_LSTM(ref_state, dt, steps_per_episode, state_vars, tracked_states, SA_noise_bias, SA_delay, incremental, training, servo_RL, servo_TF);

    %% Instantiate the SAC framework
    %sac = SAC_parallel(env);
    %sac = SAC_parallel_for_LSTM_or_FF(env, "FF"); % train with feedforward actor and critic
    sac = SAC_parallel_for_LSTM_or_FF(env, "LSTM"); % train with LSTM actor and critic
    
    %% Choose hyperparameters and train
    total_steps = 3.2e5; % training duration in number of time steps
    buffer_size = 3.6e4; % replay buffer size in time steps
    minibatch_size = 64; % number of samples in the minibatch
    n_par_envs = 6; % number of parallel environments. The limit is currently 6 even if more cores are available
    num_epoch = 1; % number of epochs to perform at each update cycle
    start_steps = 0; % can be used to choose actions at random instead of evaluating the policy at the beginning of training
    update_freq = steps_per_episode; % number of time steps between updates. It does not compensate for parallel environments
    update_after = steps_per_episode; % steps before initiating the first update cycle. Can be used to fill up the replay buffer
    lr_init = 1e-2; % initial learning rate
    lr_final = 1e-5; % final learning rate. Set to lr_init for a constant learning rate
    gamma = 0.99; % discount factor
    auto_alpha = true; % automatic temperature coefficient. if set to false, alpha=0.2
    polyak = 5e-3; % polyak update coefficient
    

    sac.train(total_steps,...
                steps_per_episode,...
                buffer_size,...
                minibatch_size,...
                num_epoch,...
                start_steps,...
                update_freq,...
                update_after,...
                lr_init,...
                lr_final,...
                gamma,...
                auto_alpha,...
                polyak,...
                stop_condition,...
                seed,...
                n_par_envs)
end
