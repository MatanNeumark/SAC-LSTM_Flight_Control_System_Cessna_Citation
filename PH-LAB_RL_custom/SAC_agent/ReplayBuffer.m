classdef ReplayBuffer < handle
    % The replay buffer supports FF or LSTM networks in the following way:
    % FF:   A minibatch is sampled randomly from the entire pool of
    %       unsampled time steps
    % LSTM: To train LSTM nets, the samples must be sequential, which 
    %       complicates things.
    %       The buffer is arranged into episodes, which are then split into
    %       multiple minibatches (so the floor of episode_length /
    %       minibatch_size). Because there is can be a non discrete number
    %       of minibatches in an episode, the minibatch is appended with
    %       previous time steps to ensure it has a minibatch_size number of
    %       samples.
    properties
        buffer_size {mustBeReal, mustBeInteger, mustBePositive}
        state_space_size {mustBeReal, mustBeInteger, mustBePositive}
        action_space_size {mustBeReal, mustBeInteger, mustBePositive}
        is_sampled
        state
        action
        reward
        new_state
        done
        terminated
        non_sampled_idx
        net_type
    end
    methods
        function obj = ReplayBuffer(buffer_size, state_space_size, action_space_size, net_type)
            % usage example: mybuff = ReplayBuffer(10, 2, 3)
            obj.buffer_size = buffer_size; % buffer capacity
            obj.state_space_size = state_space_size; % a discrete number
            obj.action_space_size = action_space_size; % a discrete number
            obj.net_type = net_type; % "LSTM" or "FF"

            obj.state = zeros(obj.buffer_size, obj.state_space_size);
            obj.action = zeros(obj.buffer_size, obj.action_space_size);
            obj.reward = zeros(obj.buffer_size, 1);
            obj.new_state = zeros(obj.buffer_size, obj.state_space_size);
            obj.done = zeros(obj.buffer_size, 1); % "1" if the agent exceeded the state bounds, or if the number of time steps per episode is reached
            obj.terminated = zeros(obj.buffer_size, 1); % "1" if the agent exceeded the state bounds
            obj.is_sampled = zeros(obj.buffer_size, 1); % a flag to keep track of which samples have already been used in the epoch
        end
    
        function store(obj, state, desired_action, reward, new_state, terminated, done)
            n_samples = length(reward);
            % usage example: mybuff = ReplayBuffer.store([1,2], [5,6,7], 3.5, [8,9])
            obj.state(end+1:end+n_samples,:) = state;
            obj.action(end+1:end+n_samples,:) = desired_action;
            obj.reward(end+1:end+n_samples,:) = reward;
            obj.new_state(end+1:end+n_samples,:) = new_state;
            obj.done(end+1:end+n_samples,:) = double(done);
            obj.terminated(end+1:end+n_samples,:) = double(terminated);
            obj.is_sampled(end+1:end+n_samples,:) = 0;

            % remove the oldest values
            obj.state(1:n_samples,:) = [];
            obj.action(1:n_samples,:) = [];
            obj.reward(1:n_samples,:) = [];
            obj.new_state(1:n_samples,:) = [];
            obj.done(1:n_samples,:) = [];
            obj.terminated(1:n_samples,:) = [];
            obj.is_sampled(1:n_samples,:) = [];
        end

        function minibatch = sample(obj, minibatch_size, t)
            % usage example: minibatch = sample(mybuff, batch_size)

            if t < minibatch_size
                error('The requested batch size is larger than the number of passed time steps. ' + ...
                    "Please set the 'update_after' variable to be larger than 'minibatch_size'")
            end
            if numel(find(obj.is_sampled==0)) < minibatch_size 
                error("The requested minibatch size is larger than the number of unsampled experiences. " + ...
                    "Reduce the number of epochs or minibatch size such that their product is smaller than " + ...
                    "the number of time steps per episode")
            end

            batch_lim = min(t, obj.buffer_size);

            if obj.net_type == "LSTM"
                filled_slots = 1 : obj.buffer_size;
                obj.non_sampled_idx = [];
                dones = [obj.buffer_size - batch_lim; find(obj.done == 1)];
                ep_lengths = dones(2:end)- dones(1:end-1);
                num_ep = length(dones) - 1;
                counter = [];
                while isempty(obj.non_sampled_idx)
                    ep = randi(num_ep);
                    obj.non_sampled_idx = filled_slots(dones(ep)+1:dones(ep+1)); % all steps in a certain episode
                    obj.non_sampled_idx = obj.non_sampled_idx(~ obj.is_sampled(dones(ep)+1:dones(ep+1))); % removes the sampled samples
                    counter(end+1) = ep;
                    if all(ismember(1:num_ep, counter)) && isempty(obj.non_sampled_idx)
                        error("The requested minibatch size is larger than the number of unsampled experiences. " + ...
                            "Reduce the number of epochs or minibatch size such that their product is smaller than " + ...
                            "the number of time steps per episode")
                    end
                end
                ep_idx = obj.non_sampled_idx(obj.non_sampled_idx~=0);
                n_minibatchs = ceil(length(ep_idx)/minibatch_size);
                remeinder = n_minibatchs*minibatch_size - length(ep_idx);
                ep_idx = [ep_idx, zeros(1, remeinder)];
                ep_idx = reshape(ep_idx, minibatch_size, n_minibatchs)';
                minibatch_idx = randi(n_minibatchs);
                idx = ep_idx(minibatch_idx,:);
                first_zero = find(idx==0, 1);
                if ~ isempty(first_zero) && ep_lengths(ep) >= minibatch_size
                    %disp("had_zeros at idx " + num2str(first_zero))
                    past_samples = idx(1) - minibatch_size + first_zero -1 : idx(1)-1;
                    non_zero_idx = idx(1:first_zero-1);
                    idx = [past_samples, non_zero_idx];
                    obj.is_sampled(non_zero_idx) = 1;
                else
                    idx = idx(idx~=0);
                    obj.is_sampled(idx) = 1;
                end

            elseif obj.net_type == "FF"
                filled_slots = (obj.buffer_size - batch_lim) + 1 : obj.buffer_size;
                obj.non_sampled_idx = filled_slots(~ obj.is_sampled(filled_slots));
                if length(obj.non_sampled_idx) < minibatch_size
                    error("The requested minibatch size is larger than the number of unsampled experiences. " + ...
                        "Reduce the number of epochs or minibatch size such that their product is smaller than " + ...
                        "the number of time steps per episode")
                end
                idx = randsample(obj.non_sampled_idx, minibatch_size);
                obj.is_sampled(idx) = 1;
            else
                error("The variable 'net_type' can take one of two values: 'LSTM' or 'FF' (feed-forward)")
            end

            if numel(find(obj.done(idx))) > 1 && obj.net_type == "LSTM"
                disp(idx)
                error("The reply buffer still doesn't work you muppet! (There was more than one 'done' in the minibatch)")
            end
            if ~all(diff(idx) == 1) && obj.net_type == "LSTM"
                disp(idx)
                error("The reply buffer still doesn't work you muppet! (Not all indices in the minibatch were consecutive)")
            end
            minibatch.state = obj.state(idx,:);
            minibatch.action = obj.action(idx,:);
            minibatch.reward = obj.reward(idx,:);
            minibatch.new_state = obj.new_state(idx,:);
            minibatch.done = obj.done(idx,:);
            minibatch.terminated = obj.terminated(idx,:);
            minibatch.sample_idx = idx;
        end
    end
end
