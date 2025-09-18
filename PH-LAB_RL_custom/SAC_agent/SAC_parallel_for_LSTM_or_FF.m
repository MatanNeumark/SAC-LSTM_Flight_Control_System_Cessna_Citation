classdef SAC_parallel_for_LSTM_or_FF < handle
    %% Implementation of the Soft Actor-Critic (SAC) reinforcement learning algorithm with support of LSTM networks.
    %% Code is written in consultation with various sources including MATLAB, OpenAI Spinning up, K. Dally github repo, Stable Baselines3, the original paper by Haarnoja: arXiv 1812.05905 and others.
    %% The thesis report and results can be found in https://repository.tudelft.nl/record/uuid:f68c711d-29f2-4c92-915b-621fa9a66026
    %% Developed by Matan Neumark, Aerospace Engineering MSc student at the Delft University of Technology. November 2024 to August 2025.

    properties
        buffer_size
        minibatch_size
        start_steps
        update_freq
        update_after
        lr
        lr_discount
        lr_init
        lr_final
        gamma
        auto_alpha
        polyak
        stop_condition
        steps_per_episode
        total_steps
        num_minibatch_per_epoch
        num_epoch
        iter_count
        epoch_count
        eval_ep_count
        ep_count
        save_freq
        eval_ep_freq
        alpha
        alpha_lr
        alpha_init
        log_alpha
        target_entropy
        env
        n_par_envs
        mdl
        bounds
        n_neurons
        buffer
        minibatch
        datalogger
        actor
        critic
        policy_averageGrad
        Q1_averageGrad
        Q2_averageGrad
        log_alpha_averageGrad
        policy_averageSqGrad
        Q1_averageSqGrad
        Q2_averageSqGrad
        log_alpha_averageSqGrad
        monitor
        initialized
        net_type
        t
    end

    methods

        function obj = SAC_parallel_for_LSTM_or_FF(env, net_type)
            obj.env = env;
            obj.net_type = net_type;
            obj.bounds = obj.env.bounds; % state bounds as given by the environment
            I_state = eye(obj.bounds.state_space_size);
            I_action = eye(obj.bounds.action_space_size);
            obj.bounds.states_upper_bounds_mat = I_state.*obj.bounds.states_upper_bounds; % transforming the upper state bounds to a matrix
            obj.bounds.states_lower_bounds_mat = I_state.*obj.bounds.states_lower_bounds; % transforming the lower state bounds to a matrix
            obj.bounds.actions_upper_bound_mat = I_action.*obj.bounds.actions_upper_bound; % transforming the upper action bounds to a matrix
            obj.bounds.actions_lower_bound_mat = I_action.*obj.bounds.actions_lower_bound; % transforming the lower action bounds to a matrix

            obj.n_neurons = 64; % number of hidden units in the networks of the actor and critic

            obj.actor = Actor_v2_LSTM_or_FF(obj.bounds, obj.n_neurons); % instantiate the actor
            obj.actor.policy = obj.actor.create_policy_net(obj.net_type); % create the policy network: LSTM of Feedforward

            obj.critic = Critic_v2_LSTM_or_FF(obj.bounds, obj.n_neurons); % instantiate the critic
            obj.critic.Q1 = obj.critic.create_Q_net(obj.net_type); % create the first Q function network: LSTM of Feedforward
            obj.critic.Q2 = obj.critic.create_Q_net(obj.net_type); % create the second Q function network: must be the same type as above
            obj.critic.Q1_targ = obj.critic.create_Q_net(obj.net_type); % create the first target Q function network: must be the same type as above
            obj.critic.Q2_targ = obj.critic.create_Q_net(obj.net_type); % create the second target Q function network: must be the same type as above

            obj.critic.Q1_targ.net.Learnables = obj.critic.Q1.net.Learnables; % clone the weights of the first Q function network
            obj.critic.Q2_targ.net.Learnables = obj.critic.Q2.net.Learnables; % clone the weights of the second Q function network

            obj.initialized = false;
        end
        


        function train(obj, total_steps, steps_per_episode, buffer_size, ...
                minibatch_size,num_epoch, start_steps, update_freq, update_after, ...
                lr_init, lr_final, gamma, auto_alpha, polyak, stop_condition, seed, n_par_envs)
            arguments
                obj 
                total_steps {mustBeInteger, mustBePositive}
                steps_per_episode {mustBeInteger, mustBePositive}
                buffer_size {mustBeInteger, mustBePositive}
                minibatch_size {mustBeInteger, mustBePositive}
                num_epoch {mustBeInteger, mustBePositive}
                start_steps {mustBeInteger, mustBeNonnegative}
                update_freq {mustBeInteger, mustBePositive}
                update_after {mustBeInteger, mustBePositive}
                lr_init {mustBePositive}
                lr_final {mustBePositive}
                gamma {mustBePositive}
                auto_alpha {mustBeA(auto_alpha, 'logical')}
                polyak {mustBePositive}
                stop_condition
                seed {mustBeInteger, mustBePositive}
                n_par_envs {mustBeInteger, mustBePositive}
            end
            obj.total_steps = total_steps; % training duration in number of time steps
            obj.lr_init = lr_init; % initial learning rate
            obj.lr_final = lr_final; % final learning rate
            obj.gamma = gamma; % discount factor
            obj.polyak = polyak; % polyak update coefficient
            obj.lr_discount = (obj.lr_final/obj.lr_init)^(1/obj.total_steps); % formula for calculating the learning rate discount factor based on the initial and final values.
            
            if obj.initialized == false % the variable 'initialized' is used to enable pausing and resuming the training of existing agents 
                obj.initialized = true;
                obj.steps_per_episode = steps_per_episode; % number of time steps per episode 
                obj.buffer_size = buffer_size; % replay buffer size in time steps
                obj.minibatch_size = minibatch_size; % number of samples in the minibatch
                obj.start_steps = start_steps; % can be used to choose actions at random instead of evaluating the policy at the beginning of training
                obj.update_freq = update_freq; % number of time steps between updates. It does not compensate for parallel environments
                obj.update_after = update_after; % steps before initiating the first update cycle. Can be used to fill up the replay buffer
                obj.auto_alpha = auto_alpha; % automatic temperature coefficient
                obj.stop_condition = stop_condition; % mean reward value to stop training. value=0 means no stop condition
                obj.n_par_envs = n_par_envs;  % number of parallel environments. The limit is currently 6 even if more cores are available
                
                obj.buffer = ReplayBuffer(obj.buffer_size, ...
                    (obj.bounds.state_space_size + obj.bounds.action_space_size), ...
                    obj.bounds.action_space_size, ...
                obj.net_type); % instantiate the replay buffer

                % the target entropy is set to the negative of the action space size. explanation can be found in the link below
                %https://stats.stackexchange.com/questions/561624/choosing-target-entropy-for-soft-actor-critic-sac-algorithm
                obj.target_entropy = -obj.bounds.action_space_size;
                obj.alpha_lr = 3e-4; % temperature coefficient learning rate if auto_alpha=true
                obj.alpha = 0.2; % only used if auto_alpha == false
                obj.alpha_init = dlarray(1); % initial value as done by K. Dally in sac.py
                obj.log_alpha = log(obj.alpha_init);
                
                % initialise arrays to track the gradients and squared gradients
                obj.policy_averageGrad = [];
                obj.Q1_averageGrad = [];
                obj.Q2_averageGrad = [];
                obj.log_alpha_averageGrad = [];
                obj.policy_averageSqGrad = [];
                obj.Q1_averageSqGrad = [];
                obj.Q2_averageSqGrad = [];
                obj.log_alpha_averageSqGrad = [];

                obj.num_minibatch_per_epoch = floor(obj.buffer_size / obj.minibatch_size); % number of minibatchs that fit in the desired number of epochs
                obj.num_epoch = num_epoch; % number of epochs to perform at each update cycle
                obj.iter_count = 0; % counter to track the number of gradient steps
                obj.epoch_count = 0; % counter to track the number of epochs performed
                obj.eval_ep_count = 0; % counter to track the number of evaluation episodes performed
                obj.save_freq = obj.total_steps; % saving frequency in time steps
                obj.eval_ep_freq = 20; % frequency to perform an evaluation episode (in number of training episodes)
                obj.datalogger.bounds = obj.bounds; % log the environment bounds

                % log the settings the the datalogger
                obj.datalogger.SAC_settings = dictionary( "total_steps", obj.total_steps,...
                                                "steps_per_episode", obj.steps_per_episode,...
                                                "buffer_zise", obj.buffer_size,...
                                                "minibatch_size", obj.minibatch_size,...
                                                "start_steps", obj.start_steps,...
                                                "update_freq", obj.update_freq,...
                                                "updae_after", obj.update_after,...
                                                "save_freq", obj.save_freq,...
                                                "eval_ep_freq", obj.eval_ep_freq,...
                                                "lr_init", obj.lr_init,...
                                                "lr_final", obj.lr_final,...
                                                "gamma", obj.gamma,...
                                                "auto_alpha", obj.auto_alpha,...
                                                "target_entropy", obj.target_entropy,...
                                                "alpha_init", obj.alpha_init,...
                                                "alpha_lr", obj.alpha_lr,...
                                                "alpha", obj.alpha,...
                                                "polyak", obj.polyak,...
                                                "num_minibatch_per_epoch", obj.num_minibatch_per_epoch,...
                                                "num_epoch", obj.num_epoch,...
                                                "n_neurons", obj.n_neurons,...
                                                "seed", seed,...
                                                "n_par_envs", obj.n_par_envs);

                agents = strings(1,obj.n_par_envs); % number of parallel environments (so not really individual 'agents')
                for n = 1:obj.n_par_envs
                    agents(n) = "Agent_" + n;
                end

                % create a training progress monitor
                obj.monitor = trainingProgressMonitor;
                obj.monitor.Metrics=[agents, "Policy_loss", "Q1_loss", "Q2_loss"];
                obj.monitor.Info=[ "Cumulative_time", "Episode", "alpha", "Mean_return", "Mean_number_of_steps", "Evaluation_return", "lr"];
                obj.monitor.XLabel="Time steps";
                groupSubPlot(obj.monitor, "Critic_loss", ["Q1_loss", "Q2_loss"]);
                yscale(obj.monitor,"Critic_loss","log")
                groupSubPlot(obj.monitor, "Return", agents);
                obj.ep_count = 0;
                obj.t = 0;

            else % if an agent is reopened for further training, the training monitor will open up
                obj.datalogger.SAC_settings("toal_steps") = obj.total_steps;
                obj.monitor.Visible = 1;
            end

            par_buffer = cell(1, obj.n_par_envs); % cell to track parallel replay buffers (if parallel environments is enabled)
            par_env = cell(1, obj.n_par_envs); % cell to track the parallel environment
            par_actor = cell(1, obj.n_par_envs); % cell to track the cloned actors to support training in parallel environments
            
            for n = 1:obj.n_par_envs
                par_buffer{n} = ReplayBuffer(obj.steps_per_episode, ...
                                (obj.bounds.state_space_size + obj.bounds.action_space_size), ...
                                obj.bounds.action_space_size, ...
                                obj.net_type); % instantiate as many replay buffers as the number of parallel environments
                par_env{n} = obj.env; % clone the environment
            end
            
            normalize_hdl = @obj.normalize;
            scale_action_hdl = @obj.scale_action;
            %unscale_action_hdl = @obj.unscale_action;

            while obj.t < obj.total_steps
                tic;
                obj.lr = obj.lr_discount^obj.t * obj.lr_init; % calculate the learning rate
                obj.ep_count = obj.ep_count + 1; % keep track of the number of episodes
                ep_return = zeros(1, n_par_envs); % keep track of the return. used in the training monitor
                ep_length = zeros(1, n_par_envs); % keep track of the length of each episode (in time steps)

                for n = 1:obj.n_par_envs
                    par_actor{n} = obj.actor; % clone the actor as many times as there are parallel environments
                end

                par_t = obj.t;
                parfor n = 1:obj.n_par_envs % parallel for loop
                    temp_buffer{n} = []
                    [state, action] = par_env{n}.reset; % reset each environment
                    done = false;
                    while ~done
                        norm_state = normalize_hdl(state); % normalise the state variables using the environment bounds
                        norm_state = [norm_state, action] % append the action (already normalised) to the state

                        if par_t >= start_steps
                            [action, ~, ~, hs] = par_actor{n}.policy_eval(norm_state, false, true, par_actor{n}.policy.net); % evaluate the policy to get a new action
                            par_actor{n}.policy.net.State = hs; % update the hidden state of the net
                            action = extractdata(action)' % extract the dl array data
                        else
                            % To fill up the buffer, random actions are sampled until t > start_steps
                            action = par_actor{n}.sample_action;
                        end

                        [new_state, reward, terminated, ~, ~, ~] = par_env{n}.step(scale_action_hdl(action)); % step in the environment
                        norm_new_state = normalize_hdl(new_state); % normalise the new state using the environment bounds
                        %applied_action = unscale_action_hdl(applied_action)
                        %norm_new_state = [norm_new_state, applied_action]
                        norm_new_state = [norm_new_state, action] % append the action (already normalised) to the state
                        ep_return(n) = ep_return(n) + reward; % for data logging
                        ep_length(n) = ep_length(n) + 1; % for data logging

                        % store sequence in the buffer
                        if terminated || ep_length(n) >= steps_per_episode
                            sprintf('env %d is done after %d steps', n, ep_length(n))
                            done = true;
                            par_buffer{n}.store(norm_state, action, reward, norm_new_state , terminated, done) % store in the replay buffer of each parallel environment
                        else
                            par_buffer{n}.store(norm_state, action, reward, norm_new_state , terminated, done)
                            state = new_state;
                        end
                    end
                    %if ep_length(n) >= minibatch_size % do not store episodes shorter than minibatch size
                        temp_buffer{n} = par_buffer{n}.sample(ep_length(n), ep_length(n)); % sample and store all experiences in a temporary buffer that can be used outside of the 'parfor' loop
                    %end
                end

                obj.datalogger.ep_return(obj.ep_count,:) = ep_return; % log in datalogger
                obj.datalogger.ep_length(obj.ep_count,:) = ep_length; % log in datalogger
                obj.datalogger.ep_count = obj.ep_count; % log in datalogger
                obj.datalogger.elapsed_time_steps(obj.ep_count) = obj.t; % log in datalogger
                obj.datalogger.ep_timestemp(obj.ep_count, :) = sum(obj.datalogger.ep_length, 1); % log in datalogger

                for n = 1:obj.n_par_envs
                    obj.buffer.store(temp_buffer{n}.state, temp_buffer{n}.action, temp_buffer{n}.reward, temp_buffer{n}.new_state, temp_buffer{n}.terminated, temp_buffer{n}.done) % store all experiences in the shared replay buffer
                end

                obj.t = obj.t + sum(ep_length); % update the current time
                obj.update_monitor(ep_return, ep_length, obj.ep_count, obj.lr, obj.t) % update the training monitor                
                if obj.ep_count == 1 % gives the progress monitor time to open before proceeding
                    pause(3)
                else
                    pause(1) % gives the progress monitor time to update the visuals before proceeding
                end
                
                % update the policy, critics and temp coefficient 
                if obj.t >= obj.update_after %&& rem(t,obj.update_freq) == 0
                    disp('time to update')
                    obj.num_minibatch_per_epoch = floor(min(obj.t, obj.buffer_size) / minibatch_size); % calculate the number of minibatchs to update with.
                    for epoch = 1:obj.num_epoch
                        obj.epoch_count = obj.epoch_count + 1;
                        for iteration = 1:obj.num_minibatch_per_epoch
                            iteration
                            obj.minibatch = obj.buffer.sample(obj.minibatch_size, obj.t); % sample a minibatch from the buffer
                            %if length(obj.minibatch.reward) >= obj.minibatch_size
                            obj.iter_count = obj.iter_count + 1;

                                [Q1_loss, Q2_loss] = obj.update_critic; % update the critics
                                policy_loss = obj.update_actor; % update the actor
    
                                if obj.auto_alpha == true
                                    log_alpha_loss = obj.update_alpha; % update the temperature coefficient

                                    obj.datalogger.log_alpha_loss(obj.iter_count) = log_alpha_loss; % log in datalogger
                                    obj.datalogger.alpha(obj.iter_count) = obj.alpha; % log in datalogger
                                end
                                obj.datalogger.Q1_loss(obj.iter_count) = Q1_loss; % log in datalogger
                                obj.datalogger.Q2_loss(obj.iter_count) = Q2_loss; % log in datalogger
                                obj.datalogger.policy_loss(obj.iter_count) = policy_loss; % log in datalogger
                            %end
                        end
                        obj.buffer.is_sampled(:) = 0; % reset the 'is_sampled' flag so the experiences can be used again in the next epoch
                        obj.datalogger.actor.policy.net(obj.epoch_count) = obj.actor.policy.net; % log the current policy in datalogger. SUPER useful because the best policy can be found in post so we don't need to worry about when to stop training
                        obj.datalogger.critic(obj.epoch_count) = obj.critic; % log in datalogger
                        obj.datalogger.time_of_update(obj.epoch_count) = obj.t; % log in datalogger
                        mean_policy_loss = mean(obj.datalogger.policy_loss(obj.iter_count - obj.num_minibatch_per_epoch + 1:obj.iter_count)); % calculate statistics for the training monitor
                        mean_Q1_loss = mean(obj.datalogger.Q1_loss(obj.iter_count - obj.num_minibatch_per_epoch + 1:obj.iter_count));  % calculate statistics for the training monitor
                        mean_Q2_loss = mean(obj.datalogger.Q2_loss(obj.iter_count - obj.num_minibatch_per_epoch + 1:obj.iter_count));  % calculate statistics for the training monitor
                        recordMetrics(obj.monitor, obj.iter_count, Policy_loss=mean_policy_loss, Q1_loss=mean_Q1_loss, Q2_loss=mean_Q2_loss); % update the training monitor
                    end
                    assignin('base','SAC_agent', obj);
                end
                time = toc;
                obj.datalogger.step_clock(obj.t) = time; % log in datalogger

                if rem(obj.ep_count, obj.eval_ep_freq) == 0 % perform an evaluation episode if its time
                    obj.eval_ep_count = obj.eval_ep_count + 1; % keep track of which evaluation episode this is
                    eval_ep_return = obj.evaluate(obj.eval_ep_count); % run the evaluation episode
                    updateInfo(obj.monitor,Evaluation_return=eval_ep_return); % update the training monitor
                    %state = obj.env.reset;
                end

                if rem(obj.t, obj.save_freq) == 0 || obj.t >= obj.total_steps %|| ((t >= obj.update_after) && (mean_return(end) > obj.stop_condition)) % save the agent to a file
                    timestamp = string(datetime('now', 'Format', 'HH-mm-ss_dd-MM-yyyy'));
                    logFileName = sprintf('SAC_%s_%s_seed_%s_%s.mat', obj.net_type, num2str(obj.env.state_vars), num2str(seed), timestamp);
                    disp('Saving data file')
                    save(logFileName, "obj")
                    disp('Done saving data file')
                end
            end
            obj.env.stop % stop the environment. only relevant when training with the Simulink environment
        end
    
        function [Q1_loss, Q2_loss, Q1_grad, Q2_grad] = compute_Q_loss(obj, minibatch, policy_net, Q1_net, Q2_net, Q1_targ_net, Q2_targ_net)
            disp('computing Q loss')
            Q1_val = obj.critic.Q_eval(minibatch.state, minibatch.action, Q1_net); % evaluate Q1 with the states and actions in the minibatch
            Q2_val = obj.critic.Q_eval(minibatch.state, minibatch.action, Q2_net); % evaluate Q2 with the states and actions in the minibatch

            % target actions for each state in the minibatch
            [new_action, new_log_prob, new_state]...
                 = obj.actor.policy_eval(minibatch.new_state, false, false, policy_net);

            % target values
            Q1_targ_val = obj.critic.Q_eval(new_state, new_action, Q1_targ_net); % evaluate Q1 (target) with the new states in the minibatch and the new actions from the policy
            Q2_targ_val = obj.critic.Q_eval(new_state, new_action, Q2_targ_net); % evaluate Q2 (target) with the new states in the minibatch and the new actions from the policy

            min_Q_targ = min(Q1_targ_val, Q2_targ_val); % choose the minimum-valued target values

            %Bellman backup. eq2 in arXiv:1812.05905v2
            backup = minibatch.reward' + obj.gamma * (1 - minibatch.terminated') .*...  
                     (min_Q_targ - obj.alpha * new_log_prob);

            % value function loss. eq5 in arXiv:1812.05905v2
            Q1_loss = 0.5 * mean((Q1_val - backup).^2, 'all');
            Q2_loss = 0.5 * mean((Q2_val - backup).^2, 'all');
            Q_loss = (Q1_loss + Q2_loss);
            Q1_grad = dlgradient(Q_loss, Q1_net.Learnables); % calculate the gradients
            Q2_grad = dlgradient(Q_loss, Q2_net.Learnables); % calculate the gradients
        end

        function [policy_loss, policy_grad] = compute_policy_loss(obj, minibatch, policy_net, Q1_net, Q2_net)
            disp('computing policy loss')
            [action, log_prob, state] = obj.actor.policy_eval(minibatch.state, false, false, policy_net); % samples actions from the policy using the states in the minibatch

            Q1_val = obj.critic.Q_eval(state, action, Q1_net); % evaluate Q1 with the states in the minibatch and the actions from the policy
            Q2_val = obj.critic.Q_eval(state, action, Q2_net); % evaluate Q2 with the states in the minibatch and the actions from the policy

            min_Q = min(Q1_val, Q2_val); % choose the minimum-valued Q-function values
            % policy loss. eq7 in arXiv:1812.05905v2
            policy_loss = mean((obj.alpha * log_prob - min_Q), "all"); %from line 216 in spinningup sac.py
            policy_grad = dlgradient(policy_loss, policy_net.Learnables); % calculate the gradients

        end

        function [log_alpha_loss, log_alpha_grad] = compute_alpha_loss(obj, minibatch, policy_net, log_alpha)
            disp('computing alpha loss')
            [~, log_prob, ~] = obj.actor.policy_eval(minibatch.state, false, false, policy_net); % evaluate the policy with the states in the minibatch to get the log of the probabilities

            log_alpha_loss = - mean(log_alpha * (log_prob + obj.target_entropy)); % loss function for the temperature coefficient
            log_alpha_grad = dlgradient(log_alpha_loss, log_alpha); % calculate the gradients
        end

        function [Q1_loss, Q2_loss] = update_critic(obj)
            % compute the losses and the gradients of the Q-functions
            [Q1_loss, Q2_loss, Q1_grad, Q2_grad] = dlfeval(@obj.compute_Q_loss,...
                                                        obj.minibatch,...
                                                        obj.actor.policy.net,...
                                                        obj.critic.Q1.net,...
                                                        obj.critic.Q2.net,...
                                                        obj.critic.Q1_targ.net,...
                                                        obj.critic.Q2_targ.net);
            
            % updates the weights using the adam function
            [obj.critic.Q1.net, obj.Q1_averageGrad, obj.Q1_averageSqGrad] = ...
            adamupdate(obj.critic.Q1.net,...
                       Q1_grad,...
                       obj.Q1_averageGrad,...
                       obj.Q1_averageSqGrad,...
                       obj.iter_count,...
                       obj.lr);

            [obj.critic.Q2.net, obj.Q2_averageGrad, obj.Q2_averageSqGrad] = ...
            adamupdate(obj.critic.Q2.net,...
                       Q2_grad,...
                       obj.Q2_averageGrad,...
                       obj.Q2_averageSqGrad,...
                       obj.iter_count,...
                       obj.lr);
            
            % polyak update of the target Q-function networks
            for lay = 1:length(obj.critic.Q1_targ.net.Learnables.Value)
                obj.critic.Q1_targ.net.Learnables.Value{lay,1} = ...
                    (1 - obj.polyak) * ...
                    obj.critic.Q1_targ.net.Learnables.Value{lay,1} + ...
                    obj.polyak * obj.critic.Q1.net.Learnables.Value{lay,1};
                obj.critic.Q2_targ.net.Learnables.Value{lay,1} = ...
                    (1 - obj.polyak) * ...
                    obj.critic.Q2_targ.net.Learnables.Value{lay,1} + ...
                    obj.polyak * obj.critic.Q2.net.Learnables.Value{lay,1};
            end
        end
        
        function policy_loss = update_actor(obj)
            % compute the losses and gradients of the policy
            [policy_loss, policy_grad] = dlfeval(@obj.compute_policy_loss,...
                                                        obj.minibatch,...
                                                        obj.actor.policy.net,...
                                                        obj.critic.Q1.net,...
                                                        obj.critic.Q2.net);
            % updates the weights using the adam function
            [obj.actor.policy.net, obj.policy_averageGrad, obj.policy_averageSqGrad] = ...
            adamupdate(obj.actor.policy.net,...
                       policy_grad,...
                       obj.policy_averageGrad,...
                       obj.policy_averageSqGrad,...
                       obj.iter_count,...
                       obj.lr);
        end

        function log_alpha_loss = update_alpha(obj)
            % compute the loss the temperature coefficient
            [log_alpha_loss, log_alpha_grad] = dlfeval(@obj.compute_alpha_loss, ...
                                                                obj.minibatch,...
                                                                obj.actor.policy.net,...
                                                                obj.log_alpha);

            % updates the log of the temperature coefficient using the adam function
            [obj.log_alpha, obj.log_alpha_averageGrad, obj.log_alpha_averageSqGrad] = ...
            adamupdate(obj.log_alpha,...
                       log_alpha_grad,...
                       obj.log_alpha_averageGrad,...
                       obj.log_alpha_averageSqGrad,...
                       obj.iter_count,...
                       obj.alpha_lr);
             
            obj.alpha = exp(obj.log_alpha); % finally, exponentiating the log of the temperature coefficient
        end

        function scaled_action = scale_action(obj, action)
            % the policy returns an action squashed between -1 and 1
            % this function scales the action according to the bound of the environment
            scaled_action = (action>=0) .* (action * abs(obj.bounds.actions_upper_bound_mat))...
                          + (action< 0) .* (action * abs(obj.bounds.actions_lower_bound_mat));
        end

        function unscaled_action = unscale_action(obj, action)
            % performs the reverse of the above- it uses the bounds to bound the action between -1 and 1
            unscaled_action = (action>=0) .* (action / abs(obj.bounds.actions_upper_bound_mat))...
                          + (action< 0) .* (action / abs(obj.bounds.actions_lower_bound_mat));
        end

        function [action, hs] = predict(obj, state, last_action, deterministic, policy_net)
            % this function samples actions deterministically from the policy
            [action, ~, ~, hs] = obj.actor.policy_eval([obj.normalize(state), obj.unscale_action(last_action)], deterministic, true, policy_net);
            action = extractdata(action)';
            action = obj.scale_action(action);
        end

        function norm_state = normalize(obj, state)
            % this function normalises the states using the environment bounds
            norm_state = (state>=0) .* (state / abs(obj.bounds.states_upper_bounds_mat))...
                       + (state< 0) .* (state / abs(obj.bounds.states_lower_bounds_mat));
        end

        function eval_ep_return = evaluate(obj, eval_ep_count)
            % this function performs and evaluation episode, in which actions are samples deterministically
            [state, action] = obj.env.reset;
            obj.actor.policy.net = resetState(obj.actor.policy.net);
            eval_ep_return = 0;
            for i = 1:obj.steps_per_episode
                [action, ~, ~, hs] = obj.actor.policy_eval([obj.normalize(state), action], true, true, obj.actor.policy.net);
                obj.actor.policy.net.State = hs;
                action = extractdata(action)';
                scaled_action = obj.scale_action(action);
                % step in the environment
                [new_state, reward, ~, ~, applied_action, ~] = obj.env.step(scaled_action);
                obj.datalogger.eval_ep_log(eval_ep_count).state(i,:) = state;
                obj.datalogger.eval_ep_log(eval_ep_count).action(i,:) = scaled_action;
                obj.datalogger.eval_ep_log(eval_ep_count).applied_action(i,:) = applied_action;
                obj.datalogger.eval_ep_log(eval_ep_count).reward(i,:) = reward;
                state = new_state;
                eval_ep_return = eval_ep_return + reward;
            end
            obj.datalogger.eval_ep_log(eval_ep_count).return = eval_ep_return;
            obj.actor.policy.net = resetState(obj.actor.policy.net);

            figure % create a figure of the time response to estimate performance
            hold on
            grid on
            for i = 1:obj.bounds.state_space_size
                plot(obj.datalogger.eval_ep_log(eval_ep_count).state(:,i), LineWidth=1)
            end
            legend([obj.env.state_vars_dict(obj.env.state_vars), obj.env.tracked_states_dict(obj.env.tracked_states)])
            title("eval episode " + num2str(eval_ep_count))

            figure % create a figure of the control surface deflection
            hold on
            grid on
            for i = 1:obj.bounds.action_space_size
                plot(obj.datalogger.eval_ep_log(eval_ep_count).action(:,i), LineWidth=0.1)
                plot(obj.datalogger.eval_ep_log(eval_ep_count).applied_action(:,i), LineWidth=0.1, LineStyle=":")
            end
            action_legend = ["\delta_e", "\delta_a", "\delta_r"];
            applied_action_legend = ["applied \delta_e", "applied \delta_a", "applied \delta_r"];
            legend([action_legend(1:obj.bounds.action_space_size), applied_action_legend(1:obj.bounds.action_space_size)])
            title("eval episode " + num2str(eval_ep_count))
        end

        function update_monitor(obj, ep_return, ep_length, ep_count, lr, t)
            % this function updates the monitor as a function of the number of parallel environments.
            % this is the bottleneck for the number of environments. so if more than 6 cores are available, change this function to expand the support
            if obj.n_par_envs == 1
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,1)), Agent_1 = ep_return(1));
            elseif obj.n_par_envs == 2
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,1)), Agent_1 = ep_return(1));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,2)), Agent_2 = ep_return(2));
            elseif obj.n_par_envs == 3
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,1)), Agent_1 = ep_return(1));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,2)), Agent_2 = ep_return(2));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,3)), Agent_3 = ep_return(3));
            elseif obj.n_par_envs == 4
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,1)), Agent_1 = ep_return(1));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,2)), Agent_2 = ep_return(2));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,3)), Agent_3 = ep_return(3));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,4)), Agent_4 = ep_return(4));
            elseif obj.n_par_envs == 5
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,1)), Agent_1 = ep_return(1));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,2)), Agent_2 = ep_return(2));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,3)), Agent_3 = ep_return(3));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,4)), Agent_4 = ep_return(4));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,5)), Agent_5 = ep_return(5));
            elseif obj.n_par_envs == 6
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,1)), Agent_1 = ep_return(1));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,2)), Agent_2 = ep_return(2));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,3)), Agent_3 = ep_return(3));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,4)), Agent_4 = ep_return(4));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,5)), Agent_5 = ep_return(5));
                recordMetrics(obj.monitor, sum(obj.datalogger.ep_length(:,6)), Agent_6 = ep_return(6));
            else
                error('You are using more prallel environments than the monitor supports (6)')
            end
            updateInfo(obj.monitor, Cumulative_time = t,...
                                Episode=ep_count, ...
                                alpha=obj.alpha, ...
                                Mean_return=mean(ep_return),...
                                Mean_number_of_steps = mean(ep_length),...
                                lr = lr);
            if t > obj.total_steps
                obj.monitor.Progress = 100 * (obj.total_steps/obj.total_steps);
            else
                obj.monitor.Progress = 100 * (t/obj.total_steps);
            end
        end
    end
end

