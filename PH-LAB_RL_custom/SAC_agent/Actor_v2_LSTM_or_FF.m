classdef Actor_v2_LSTM_or_FF < handle
    properties
        bounds
        n_neurons {mustBeReal, mustBeInteger, mustBePositive}
        state_space_size
        action_space_size
        log_sigma_min
        log_sigma_max
        eps 
        policy
        net_type
    end

    methods
        function obj = Actor_v2_LSTM_or_FF(bounds, n_neurons)
            obj.bounds = bounds;
            obj.n_neurons = n_neurons;
            obj.state_space_size = obj.bounds.state_space_size;
            obj.action_space_size = obj.bounds.action_space_size;
            %obj.n_ref_states = obj.bounds.n_ref_states;
            obj.log_sigma_min = -20;
            obj.log_sigma_max = 2;
            obj.eps = 1e-6; % a small number for numerical stability  
        end

        function policy = create_policy_net(obj, net_type)
            obj.net_type = net_type;
            policy.net = dlnetwork;
            
            if obj.net_type == "LSTM"
                layers = [
                    sequenceInputLayer(obj.state_space_size + obj.action_space_size,"Name","delayed_obs_act")
                    lstmLayer(obj.n_neurons,"Name","lstm")
                    fullyConnectedLayer(obj.n_neurons,"Name","fc_2","WeightsInitializer","he")
                    reluLayer("Name","relu_2")];
                policy.net = addLayers(policy.net, layers);
    
                layers = [
                    fullyConnectedLayer(obj.action_space_size,"Name","fc_mu","WeightsInitializer","he")];
                policy.net = addLayers(policy.net, layers);
    
                layers = [
                    fullyConnectedLayer(obj.action_space_size,"Name","fc_sigma","WeightsInitializer","he")
                    softplusLayer("Name","softplus")];
                policy.net = addLayers(policy.net, layers);
 
                policy.net = connectLayers(policy.net,"relu_2","fc_mu");
                policy.net = connectLayers(policy.net,"relu_2","fc_sigma");

            elseif obj.net_type == "FF"
                layers = [
                    featureInputLayer(obj.state_space_size  + obj.action_space_size, Name="featureinput")
                    fullyConnectedLayer(obj.n_neurons, WeightsInitializer="he", Name="fc_1")
                    layerNormalizationLayer(Name='norm_1')
                    reluLayer(Name="relu_1")
                    fullyConnectedLayer(obj.n_neurons, WeightsInitializer="he", Name="fc_2")
                    layerNormalizationLayer(Name='norm_2')
                    reluLayer(Name="relu_2")];
                policy.net = addLayers(policy.net, layers);
    
                % first branch splits at the relu layer and gives mu
                layers = [
                    fullyConnectedLayer(obj.action_space_size, WeightsInitializer="he", Name="fc_mu_branch")];
                policy.net = addLayers(policy.net, layers);
                
                % second branch splits at the relu layer and gives sigma
                layers = [
                    fullyConnectedLayer(obj.action_space_size, WeightsInitializer="he", Name="fc_sigma_branch")
                    softplusLayer(Name="softplus")];
                policy.net = addLayers(policy.net, layers);
                
                % connecting the branches to relu_1
                policy.net = connectLayers(policy.net,"relu_2","fc_mu_branch");
                policy.net = connectLayers(policy.net,"relu_2","fc_sigma_branch");
            else
                error("The variable 'net_type' can take one of two values: 'LSTM' or 'FF' (feed-forward)")
            end

            clear layers;

            policy.net = initialize(policy.net);
            %figure % a figure to visualise the net
            %plot(policy.net)
        end

        function [action, log_prob, dl_state, hs] = policy_eval(obj, state, deterministic, inference, net)
            if islogical(deterministic) == false
                disp('Please input a logical value for the variable "deterministic". i.e, "true" or "false"')
            end
            % the use of GPU can be enabled by uncommenting the code below
            if obj.net_type == "LSTM"
                dl_state = dlarray(state', 'CT');
                % if canUseGPU 
                %     %disp('using GPU');
                %     dl_state = gpuArray(dl_state);
                % else
                %     %disp('not using GPU');
                % end
            else
                dl_state = dlarray(state', 'CB');
                % if canUseGPU
                %     %disp('using GPU');
                %     dl_state = gpuArray(dl_state);
                % else
                %     %disp('not using GPU');
                % end
            end 

            if inference == false
                % forward pass of the state through the net.
                % Used for training
                [mu, sigma] = forward(net, dl_state);
            else
                % predict is used for inference of a trained net or during
                % experience accumulation
                [mu, sigma, hs] = predict(net, dl_state);
                %net.State = hs;
            end
            
            % clip the log_sigma value
            %dl_clipped_log_sigma = clip(dl_log_sigma, obj.log_sigma_min, obj.log_sigma_max);

            % calculate sigma. It can ranges between 0 (e^-20) and e^2.
            %dl_sigma = exp(dl_clipped_log_sigma);

            % pre-squashed distribution and sampling
            if deterministic == false
                %disp('stochastic policy evaluation');
                % sample a stochastic action from normal distribution
                action = mu + sigma .* randn(obj.action_space_size, size(state,1));
            elseif deterministic == true
                %disp('Deterministic policy evaluation');
                action = mu;
            end
            %prob = (1./(dl_sigma*sqrt(2*pi))).*exp(sum(-0.5 * (((dl_action - dl_mu) ./ (dl_sigma + obj.eps)).^2 + 2*log(dl_sigma) + log(2*pi)),1));
            % log_prob is the log of the probability density function for a normal distribution.
            % sum across actions. output is a vector with minibatch_size number of elements.
            log_prob = sum(-0.5 * (((action - mu) ./ (sigma + obj.eps)).^2 + 2*log(sigma) + log(2*pi)),1); 
            % squashing
            [action, log_prob] = obj.squashedgaussian(action, log_prob);

            % dropping action commands if control axis is unused
            %action = action(1:obj.bounds.action_space_size,:);
            %action = extractdata(action)';
        end

        function [squashed_action, squashed_log_prob] = squashedgaussian(obj, action, log_prob)
            squashed_action = tanh(action);
            % sum across actions. output is a vector with minibatch_size number of elements.
            % the equation below is taken from eq26 in arXiv:1812.05905v2
            squashed_log_prob = log_prob - sum(log(1 - squashed_action.^2 + obj.eps),1);
        end

        function action = sample_action(obj)
            % random action sampling without evaluating the net.
            % can be useful to accelerate experience accumulation at the beginning of training
            action = zeros(1,length(obj.bounds.action_space_size));
            for i = 1:obj.bounds.action_space_size
                action(i) = 1 - (2 * rand);
            end
        end

    end
end