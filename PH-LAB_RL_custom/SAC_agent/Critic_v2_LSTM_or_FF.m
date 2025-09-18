classdef Critic_v2_LSTM_or_FF < handle
    properties
        bounds
        n_neurons {mustBeReal, mustBeInteger, mustBePositive}
        state_space_size
        action_space_size
        Q1
        Q2
        Q1_targ
        Q2_targ
        net_type
    end

    methods
        function obj = Critic_v2_LSTM_or_FF(bounds, n_neurons)
            obj.bounds = bounds;
            obj.n_neurons = n_neurons;
            obj.state_space_size = obj.bounds.state_space_size;
            obj.action_space_size = obj.bounds.action_space_size;
        end

        function Q_func = create_Q_net(obj, net_type)
            obj.net_type = net_type;
            Q_func.net = dlnetwork;
         
            if obj.net_type == "LSTM"
                layers = [
                    sequenceInputLayer(obj.state_space_size + obj.action_space_size,"Name","delayed_state_action")
                    lstmLayer(obj.n_neurons,"Name","lstm")
                    fullyConnectedLayer(obj.n_neurons,"Name","fc_2","WeightsInitializer","he")
                    reluLayer("Name","relu_2")
                    fullyConnectedLayer(1,"Name","fc_3","WeightsInitializer","he")];
                Q_func.net = addLayers(Q_func.net, layers);
                
            elseif obj.net_type == "FF"
                layers = [
                    featureInputLayer(obj.state_space_size,"Name","state_featureinput")
                    fullyConnectedLayer(obj.n_neurons, WeightsInitializer="he", Name="fc_1")];
                Q_func.net = addLayers(Q_func.net, layers);
                
                layers = [
                    featureInputLayer(obj.action_space_size, "Name","action_featureinput")
                    fullyConnectedLayer(obj.n_neurons, WeightsInitializer="he", Name="fc_2")];
                Q_func.net = addLayers(Q_func.net, layers);
                
                layers = [
                    concatenationLayer(1, 2, Name="concat")
                    layerNormalizationLayer(Name='norm_1')
                    reluLayer("Name","relu_1")
                    fullyConnectedLayer(obj.n_neurons, WeightsInitializer="he", Name="fc_3")
                    layerNormalizationLayer(Name='norm_2')
                    reluLayer("Name","relu_2")
                    fullyConnectedLayer(1, WeightsInitializer="he", Name="fc_4")];
                    Q_func.net = addLayers(Q_func.net, layers);
                
                Q_func.net = connectLayers(Q_func.net,"fc_1","concat/in1");
                Q_func.net = connectLayers(Q_func.net,"fc_2","concat/in2");
            else
                error("The variable 'net_type' can take one of two values: 'LSTM' or 'FF' (feed-forward)")
            end
            
            clear layers;

            Q_func.net = initialize(Q_func.net);
            %figure % a figure to visualise the net
            %plot(Q_func.net)
        end

        function Q_val = Q_eval(obj, state, action, net)


            if obj.net_type == "LSTM"
                if isdlarray(state) == false
                    dl_state = dlarray(state', 'CT');
                else
                    dl_state = state;
                end

                if isdlarray(action) == false
                    dl_action = dlarray(action', 'CT');
                else
                    dl_action = action;
                end

                % the use of GPU can be enabled by uncommenting the code below
                % if canUseGPU
                %     %disp('using GPU');
                %     dl_state = gpuArray(dl_state);
                %     dl_action = gpuArray(dl_action);
                % else
                %     %disp('not using GPU');
                % end
                [Q_val] = forward(net, [dl_state(1:obj.state_space_size,:); dl_action]);
            else
                if isdlarray(state) == false
                    dl_state = dlarray(state', 'CB');
                else
                    dl_state = state;
                end

                if isdlarray(action) == false
                    dl_action = dlarray(action', 'CB');
                else
                    dl_action = action;
                end
                % the use of GPU can be enabled by uncommenting the code below
                % if canUseGPU
                %     %disp('using GPU');
                %     dl_state = gpuArray(dl_state);
                %     dl_action = gpuArray(dl_action);
                % else
                %     %disp('not using GPU');
                % end
                [Q_val] = forward(net, dl_state(1:obj.state_space_size,:), dl_action);
            end
        end
    end
end