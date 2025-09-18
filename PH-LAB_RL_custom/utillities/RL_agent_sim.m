classdef RL_agent_sim < handle

    properties
        steps_per_episode
    end

    methods
        function obj = RL_agent_sim(steps_per_episode)
            obj.steps_per_episode = steps_per_episode;
        end

        function [state, action, reward, applied_action, internal_state, nMAE, ctrl_activity] = linear_env(obj, agent, policy_net, ref_state, SA_noise_bias, SA_delay, servo_RL, servo_TF)
            agent.env.ref_state = ref_state;
            agent.env.training = false;
            agent.env.SA_noise_bias = SA_noise_bias;
            agent.env.SA_delay = SA_delay;
            agent.env.servo_RL = servo_RL;
            agent.env.servo_TF = servo_TF;
            deterministic = true;
            ref_idx = length(agent.env.state_vars) + 1 : length(agent.env.state_vars) + length(agent.env.tracked_idx);
            state = [];
            action = zeros(obj.steps_per_episode, agent.bounds.action_space_size);
            reward = zeros(obj.steps_per_episode, 1);
            applied_action = zeros(obj.steps_per_episode, agent.bounds.action_space_size);
            internal_state = [];
            [state(1,:), action(1,:)] = agent.env.reset;

            %if class(agent) == "SAC_parallel_GRU"
                applied_action(1,:) = action(1,:);
                internal_state(1,:) = state(1,1:length(agent.env.state_vars));
                policy_net = resetState(policy_net);
                for t = 1:obj.steps_per_episode
                    t;
                    [action(t+1,:), hs] = agent.predict(state(t,:), action(t,:), deterministic, policy_net);
                    policy_net.State = hs;
                    [state(t+1,:), reward(t), ~, ~, applied_action(t+1,:), internal_state(t+1,:)] = agent.env.step(action(t+1,:));
                end
            % else
            %     [state(1,:), ~] = agent.env.reset;
            %     for t = 1:obj.steps_per_episode
            %         t;
            %         action(t+1,:) = agent.predict(state(t,:), deterministic, policy_net);
            %         [state(t+1,:), reward(t), ~, ~, ~, ~] = agent.env.step(action(t+1,:));
            %     end
            % end
            nMAE = mean(abs(state(:,ref_idx) - state(:,agent.env.tracked_idx))) / mean(abs(state(:,ref_idx)));
            ctrl_activity = trapz(agent.env.dt, abs(diff(action)/agent.env.dt)) / (obj.steps_per_episode*agent.env.dt);
        end


        function [state, action, reward, applied_action, internal_state, nMAE, ctrl_activity] = simulink_env(obj, agent, policy_net, ref_state, SA_noise_bias, SA_delay, servo_RL, servo_TF)
            incremental = agent.env.incremental;
            env_mdl = "Citation_RL_custom_env";
            simulink_env = simulink_citation_env(ref_state,...
                                                agent.env.dt,...
                                                obj.steps_per_episode,...
                                                agent.env.state_vars,...
                                                agent.env.tracked_states,...
                                                SA_noise_bias,...
                                                SA_delay,...
                                                incremental,...
                                                servo_RL,...
                                                servo_TF,...
                                                env_mdl);

            deterministic = true;
            ref_idx = length(agent.env.state_vars) + 1 : length(agent.env.state_vars) + length(agent.env.tracked_idx);
            state = [];
            action = [];
            reward = zeros(obj.steps_per_episode, 1);
            applied_action = [];
            internal_state = [];
            if simulink_env.sm.Status == "inactive"
                simulink_env.stop
            end
            [state(1,:), action(1,:)] = simulink_env.reset;

            % if class(agent) == "SAC_parallel_GRU"
                applied_action(1,:) = action(1,:);
                internal_state(1,:) = state(1,1:length(agent.env.state_vars));
                policy_net = resetState(policy_net);
                for t = 1:obj.steps_per_episode
                    t
                    [action(t+1,:), hs] = agent.predict(state(t,:), action(t,:), deterministic, policy_net);
                    policy_net.State = hs;
                    [state(t+1,:), reward(t), ~, ~, applied_action(t+1,:), internal_state(t+1,:)] = simulink_env.step(action(t+1,:));
                end
            % else
            %     [state(1,:), ~] = agent.env.reset;
            %     for t = 1:obj.steps_per_episode
            %         t
            %         action(t+1,:) = agent.predict(state(t,:), deterministic, policy_net);
            %         [state(t+1,:), reward(t), ~, ~, ~, ~] = simulink_env.step(action(t+1,:));
            %     end
            % end
            simulink_env.stop
            nMAE = mean(abs(state(:,ref_idx) - state(:,agent.env.tracked_idx))) / mean(abs(state(:,ref_idx)));
            ctrl_activity = trapz(agent.env.dt, abs(diff(action)/agent.env.dt)) / (obj.steps_per_episode*agent.env.dt);
        end

        function [state, action, reward, applied_action, internal_state, nMAE, ctrl_activity] = PID_in_linear_env(obj, controller, env, ref_state, SA_noise_bias, SA_delay, servo_RL, servo_TF)
            env.ref_state = ref_state;
            env.training = false;
            env.SA_noise_bias = SA_noise_bias;
            env.SA_delay = SA_delay;
            env.servo_RL = servo_RL;
            env.servo_TF = servo_TF;
            ref_idx = length(env.state_vars) + 1 : length(env.state_vars) + length(env.tracked_idx);
            state = [];
            action = [];
            reward = zeros(obj.steps_per_episode, 1);
            applied_action = [];
            internal_state = [];
            [state(1,:), action(1,:)] = env.reset;
            controller.reset
            applied_action(1,:) = action(1,:);
            internal_state(1,:) = state(1,1:length(env.state_vars));
            for t = 1:obj.steps_per_episode
                t
                action(t+1,:) = controller.control(state(t, 1), state(t, 3), state(t, 4));
                action(t+1,:) = clip(action(t+1,:), env.lo_sat_lim, env.up_sat_lim);
                [state(t+1,:), reward(t), ~, ~, applied_action(t+1), internal_state(t+1,:)] = env.step(action(t+1,:));
            end
            nMAE = mean(abs(state(:,ref_idx) - state(:,env.tracked_idx))) / mean(abs(state(:,ref_idx)));
            ctrl_activity = trapz(env.dt, abs(diff(action)/env.dt)) / (obj.steps_per_episode*env.dt);
        end

        function [state, action, reward, applied_action, internal_state, nMAE, ctrl_activity] = PID_in_simulink_env(obj, controller, env, ref_state, SA_noise_bias, SA_delay, servo_RL, servo_TF)
            incremental = env.incremental;
            env_mdl = "Citation_RL_custom_env";
            simulink_env = simulink_citation_env(ref_state,...
                                                env.dt,...
                                                obj.steps_per_episode,...
                                                env.state_vars,...
                                                env.tracked_states,...
                                                SA_noise_bias,...
                                                SA_delay,...
                                                incremental,...
                                                servo_RL,...
                                                servo_TF,...
                                                env_mdl);
            ref_idx = length(simulink_env.state_vars) + 1 : length(simulink_env.state_vars) + length(simulink_env.tracked_idx);
            state = [];
            action = [];
            reward = zeros(obj.steps_per_episode, 1);
            applied_action = [];
            internal_state = [];
            if simulink_env.sm.Status == "inactive"
                simulink_env.stop
            end
            [state(1,:), action(1,:)] = simulink_env.reset;
            controller.reset

            applied_action(1,:) = action(1,:);
            internal_state(1,:) = state(1,1:length(env.state_vars));
            for t = 1:obj.steps_per_episode
                t
                action(t+1,:) = controller.control(state(t, 1), state(t, 3), state(t, 4));
                action(t+1,:) = clip(action(t+1,:), env.lo_sat_lim, env.up_sat_lim);
                [state(t+1,:), reward(t), ~, ~, applied_action(t+1), internal_state(t+1,:)] = simulink_env.step(action(t+1,:));
            end
            simulink_env.stop
            nMAE = mean(abs(state(:,ref_idx) - state(:,env.tracked_idx))) / mean(abs(state(:,ref_idx)));
            ctrl_activity = trapz(env.dt, abs(diff(action)/env.dt)) / (obj.steps_per_episode*env.dt);

        end


        function stats = best_policy(obj, agent, ref_state, SA_noise_bias, SA_delay, servo_RL, servo_TF)
            agent.env.SA_noise_bias = SA_noise_bias;
            agent.env.SA_delay = SA_delay;
            agent.env.servo_RL = servo_RL;
            agent.env.servo_TF = servo_TF;
            agent.env.training = false;
            %tracked_idx = agent.env.tracked_idx;
            %ref_idx = length(agent.env.state_vars) + 1 : length(agent.env.state_vars) + length(tracked_idx);
            action = cell(length(agent.datalogger.actor.policy.net), 1);
            reward = cell(1, length(agent.datalogger.actor.policy.net));
            state = cell(length(agent.datalogger.actor.policy.net), 1);
            ep_return = zeros(length(agent.datalogger.actor.policy.net), 1);
            ctrl_activity = zeros(length(agent.datalogger.actor.policy.net), 1);
            nMAE = zeros(length(agent.datalogger.actor.policy.net), 1);
            for i = 1:length(agent.datalogger.actor.policy.net)
                disp("Running policy " + num2str(i) + "/" + num2str(length(agent.datalogger.actor.policy.net)))
                policy_net = agent.datalogger.actor.policy.net(i);
                [state{i}, action{i}, reward{i}, ~, ~, nMAE(i), ctrl_activity(i)] = obj.linear_env(agent, policy_net, ref_state, SA_noise_bias, SA_delay, servo_RL, servo_TF);
                ep_return(i) = sum(reward{i});
                % ctrl_activity(i) = trapz(agent.env.dt, abs(diff(action{i})/agent.env.dt)) / (obj.steps_per_episode*agent.env.dt);
                % nMAE(i) = mean(abs(state{i}(:,ref_idx) - state{i}(:,tracked_idx))) / mean(abs(state{i}(:,ref_idx)))
            end
            %[~, idx] = max(ep_return);
            [~, idx] = min(nMAE);
            best_policy = agent.datalogger.actor.policy.net(idx);

            stats.best_policy = best_policy;
            stats.idx = idx;
            stats.ep_return = ep_return;
            stats.ctrl_activity = ctrl_activity;
            stats.nMAE = nMAE;
        end

        % function [policy_net, idx] = best_policy(obj, agent)
        %     [max_return,idx] = max(agent.datalogger.ep_return ./ agent.datalogger.ep_length, [], "all");
        %     [row, ~] = ind2sub(size(agent.datalogger.ep_return), idx);
        %     idx = row - 1;
        %     policy_net = agent.datalogger.actor.policy.net(idx);
        % end

    end

end