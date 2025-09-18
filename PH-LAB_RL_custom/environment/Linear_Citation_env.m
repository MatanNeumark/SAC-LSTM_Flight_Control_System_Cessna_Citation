classdef Linear_Citation_env < rl.env.MATLABEnvironment
    properties
        state_vars_dict = dictionary(1, "p", 2, "q", 3, "r", 4, "V_TAS", 5, "alpha", 6, "beta", 7, "phi", 8, "theta", 9,  "psi", 10, "he", 11, "xe", 12, "ye")
        tracked_states_dict = dictionary(1, "p_{ref}", 2, "q_{ref}", 3, "r_{ref}", 4, "V_TAS_{ref}", 5, "alpha_{ref}", 6, "beta_{ref}", 7, "phi_{ref}", 8, "theta_{ref}", 9,  "psi_{ref}", 10, "he_{ref}", 11, "xe_{ref}", 12, "ye_{ref}")
        p = 1;
        q = 2;
        r = 3;
        V_TAS = 4;
        alpha = 5;
        beta = 6;
        phi = 7;
        theta = 8;
        psi = 9;
        h = 10;
        ref_state
        dt = 0.01
        steps_per_episode
        state_vars
        tracked_states
        S_A_dynamics
        incremental
        training
        tracked_idx
        ep_length
        bounds
        reset_up_bound
        reset_lo_bound
        A
        B
        I
        RF
        x0
        up_sat_lim
        lo_sat_lim
        delta_lim
        outvec
        yndx
        de_ss
        da_ss
        dr_ss
        servos
        x_servos
        u0
        state
        last_action;
        is_done = false
        counter = 1
        error_cost
        % de_up_sat_lim = deg2rad(15); % upper limit
        % de_lo_sat_lim = deg2rad(-17); % lower limit
        % da_sat_lim = deg2rad(22); % symmetric
        % dr_sat_lim = deg2rad(34); % symmetric
        % de_rate_lim = deg2rad(20) % limit is 20 deg/s
        % da_rate_lim = deg2rad(20) % limit is 20 deg/s
        % dr_rate_lim = deg2rad(40) % limit is 40 deg/s
        b_rates_logical
        b_rates_idx
        V_TAS_logical
        V_TAS_idx
        euler_logical
        euler_idx
        aero_logical
        aero_idx
        alt_logical
        alt_idx

        b_rate_bias = 3e-5;
        b_rate_std = 4e-7;
        b_rate_delay = 15e-3;
        %b_rate_delay = 100e-3
        b_rate_freq = 1000;
        b_rate_buffer

        V_TAS_bias = 2.5;
        V_TAS_std = 8.5e-4;
        V_TAS_delay = 300e-3;
        V_TAS_freq = 8;
        V_TAS_buffer

        euler_bias = 4e-3;
        euler_std = 1e-9;
        euler_delay = 90e-3;
        euler_freq = 52;
        euler_buffer

        aero_bias = 1.8e-3;
        aero_std = 7.5e-8;
        aero_delay = 100e-3;
        aero_freq = 100;
        aero_buffer

        alt_bias = 8e-3;
        alt_std = 4.5e-3;
        alt_delay = 300e-3;
        alt_freq = 20;
        alt_buffer

        servo_bias = 2.4e-3;
        servo_std = 5.5e-7;
        servo_delay = 10e-3;
        servo_freq = 100;
        servo_buffer
        action_hist
    end
    

    methods              
        function obj = Linear_Citation_env(ref_state, dt, steps_per_episode, state_vars, tracked_states, S_A_dynamics, incremental, training)
            Env_config;
            delta_de_lim = de_rate_lim * dt;
            delta_da_lim = da_rate_lim * dt;
            delta_dr_lim = dr_rate_lim * dt;
            % ObservationInfo = rlNumericSpec([1 12]);
            if width(ref_state) > height(ref_state)
                ref_state = ref_state';
            end
            if all(~ismember([p, r, phi, Beta], state_vars)) && min(size(ref_state)) == 1 && incremental == true % if longitudinal
                ActionInfo = rlNumericSpec([1 1], LowerLimit=-delta_de_lim, UpperLimit=delta_de_lim);
                ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states) + ActionInfo.Dimension(2)]);
                ActionInfo.Name = 'Elevator';           
            elseif all(~ismember([p, r, phi, Beta], state_vars)) && min(size(ref_state)) == 1 && incremental == false
                ActionInfo = rlNumericSpec([1 1], LowerLimit=de_lo_sat_lim, UpperLimit=de_up_sat_lim);
                ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states)]);
                ActionInfo.Name = 'Elevator';           
            elseif all(ismember([p, q, r, phi, Beta], state_vars)) && min(size(ref_state)) == 3 && incremental == true % if longitudinal and lateral
                ActionInfo = rlNumericSpec([1 3], LowerLimit=[-delta_de_lim, -delta_da_lim, -delta_dr_lim], UpperLimit=[delta_de_lim, delta_da_lim, delta_dr_lim]);
                ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states) + ActionInfo.Dimension(2)]);
                ActionInfo.Name = 'Elevator, Aileron, Rudder';           
            elseif all(ismember([p, q, r, phi, Beta], state_vars)) && min(size(ref_state)) == 3 && incremental == false
                ActionInfo = rlNumericSpec([1 3], LowerLimit=[de_lo_sat_lim, de_lo_sat_lim, de_lo_sat_lim], UpperLimit=[de_up_sat_lim, de_up_sat_lim, de_up_sat_lim]);
                ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states)]);
                ActionInfo.Name = 'Elevator, Aileron, Rudder';           
            else
                error("Please define a valid set of state variables and reference signals")
            end
            ObservationInfo.Name = 'Citation States';
            ObservationInfo.Description = num2str(state_vars);
            % The following line implements built-in functions of RL env
            obj = obj@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);

            obj.ref_state = ref_state;            
            obj.dt = dt;
            obj.steps_per_episode = steps_per_episode;
            obj.state_vars = state_vars;
            obj.tracked_states = tracked_states;
            obj.S_A_dynamics = S_A_dynamics;
            obj.incremental = incremental;
            obj.training =training;
            obj.ep_length = 0:obj.dt:obj.dt*obj.steps_per_episode;

            obj.bounds.state_space_size = ObservationInfo.Dimension(2); % alpha, q
            obj.bounds.action_space_size = ActionInfo.Dimension(2); % de
            %load Matan_v90ms_m4500kg_h2000m_q_init.lin -mat
            load('environment/full_DoF_v90ms_m4500kg_h2000m_q_init.lin', '-mat')
            obj.up_sat_lim = [de_up_sat_lim, da_sat_lim, dr_sat_lim];
            obj.up_sat_lim = obj.up_sat_lim(1:obj.bounds.action_space_size);
            obj.lo_sat_lim = [de_lo_sat_lim, -da_sat_lim, -dr_sat_lim];
            obj.lo_sat_lim = obj.lo_sat_lim(1:obj.bounds.action_space_size);
            obj.delta_lim = [delta_de_lim, delta_da_lim, delta_dr_lim];
            obj.delta_lim = obj.delta_lim(1:obj.bounds.action_space_size);
            obj.u0 = u0;
            [~, obj.tracked_idx] = ismember(obj.tracked_states, obj.state_vars);
            obj.outvec = outvec(obj.state_vars,:);
            %obj.yndx = yndx;
            obj.A = Alin(obj.state_vars,obj.state_vars);
            obj.B = Blin(obj.state_vars,1:ActionInfo.Dimension(2));
            % obj.x0 = [x0(obj.yndx), obj.ref_state(1,:)];
            obj.x0 = x0;
            obj.x0([obj.alpha, obj.beta, obj.theta]) = 0;
            obj.x0 = obj.x0(obj.state_vars);
            obj.de_ss = ss(tf(13, [1 13]));
            obj.da_ss = ss(tf(13, [1 13]));
            obj.dr_ss = ss(tf(13, [1 13]));
            obj.servos.A = [obj.de_ss.A; obj.da_ss.A; obj.dr_ss.A];
            obj.servos.A = obj.servos.A(1:obj.bounds.action_space_size); 
            obj.servos.B = [obj.de_ss.B; obj.da_ss.B; obj.dr_ss.B];
            obj.servos.B = obj.servos.B(1:obj.bounds.action_space_size); 
            obj.servos.C = [obj.de_ss.C; obj.da_ss.C; obj.dr_ss.C];
            obj.servos.C = obj.servos.C(1:obj.bounds.action_space_size); 
            obj.servos.D = [obj.de_ss.D; obj.da_ss.D; obj.dr_ss.D];
            obj.servos.D = obj.servos.D(1:obj.bounds.action_space_size); 
            % obj.bounds.states_upper_bounds = [deg2rad(20), deg2rad(20), deg2rad(20), 130, deg2rad(20), deg2rad(20), deg2rad(20), deg2rad(20), 2300, deg2rad(20), deg2rad(20), deg2rad(20)];
            % obj.bounds.states_lower_bounds = [-deg2rad(20), -deg2rad(20), -deg2rad(20), 50, -deg2rad(20), -deg2rad(20), -deg2rad(20), -deg2rad(20), 1700, -deg2rad(20), -deg2rad(20), -deg2rad(20)];
            obj.bounds.states_upper_bounds = [deg2rad(40), deg2rad(20), deg2rad(20), 130, deg2rad(20), deg2rad(20), deg2rad(20), deg2rad(20), inf, 2300, inf, inf];
            obj.bounds.states_lower_bounds = [-deg2rad(40), -deg2rad(20), -deg2rad(20), 50, -deg2rad(20), -deg2rad(20), -deg2rad(20), -deg2rad(20), -inf, 1700, -inf, -inf];
            if incremental == true
                obj.bounds.states_upper_bounds = [obj.bounds.states_upper_bounds(obj.state_vars), obj.bounds.states_upper_bounds(obj.tracked_states), obj.up_sat_lim(1:ActionInfo.Dimension(2))];
                obj.bounds.states_lower_bounds = [obj.bounds.states_lower_bounds(obj.state_vars), obj.bounds.states_lower_bounds(obj.tracked_states), obj.lo_sat_lim(1:ActionInfo.Dimension(2))];
            else
                obj.bounds.states_upper_bounds = [obj.bounds.states_upper_bounds(obj.state_vars), obj.bounds.states_upper_bounds(obj.tracked_states)];
                obj.bounds.states_lower_bounds = [obj.bounds.states_lower_bounds(obj.state_vars), obj.bounds.states_lower_bounds(obj.tracked_states)];
            end
            obj.bounds.actions_upper_bound = ActionInfo.UpperLimit;
            obj.bounds.actions_lower_bound = ActionInfo.LowerLimit;
            obj.reset_up_bound = 1e-2 * obj.bounds.states_upper_bounds;
            obj.reset_lo_bound = 1e-2 * obj.bounds.states_lower_bounds;
            obj.I = eye(length(obj.state_vars));
            obj.error_cost = [1, 1, 1, 1, 1, 4, 1, 1, 1, 1];
            obj.error_cost = obj.error_cost(obj.tracked_states);
            obj.error_cost = obj.error_cost/sum(obj.error_cost);
            [obj.b_rates_logical, obj.b_rates_idx] = ismember([obj.p, obj.q, obj.r], obj.state_vars);
            [obj.V_TAS_logical, obj.V_TAS_idx] = ismember(obj.V_TAS, obj.state_vars);
            [obj.euler_logical, obj.euler_idx] = ismember([obj.phi, obj.theta, obj.psi], obj.state_vars);
            [obj.aero_logical, obj.aero_idx] = ismember([obj.alpha, obj.beta], obj.state_vars);
            [obj.alt_logical, obj.alt_idx] = ismember(obj.h, obj.state_vars);



        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [new_state,reward,is_done,Info, internal_state] = step(obj, delta_action)
            Info = [];
            if obj.incremental == true && obj.S_A_dynamics == true
                    action = obj.servo_dynamics(delta_action);
                    s = ((obj.A*obj.dt + obj.I)*obj.state' + obj.B*obj.dt*action')';
                    obj.state = s;
                    s = obj.sensor_dynamics(s);
                    new_state = [s, obj.ref_state(obj.counter+1,:), action];

            elseif obj.incremental == false && obj.S_A_dynamics == true
                    action = delta_action;
                    action = obj.servo_dynamics(action);
                    s = ((obj.A*obj.dt + obj.I)*obj.state' + obj.B*obj.dt*action')';
                    obj.state = s;
                    s = obj.sensor_dynamics(s);
                    new_state = [s, obj.ref_state(obj.counter+1,:)];
            elseif obj.incremental == true && obj.S_A_dynamics == false
                    action = obj.servo_dynamics(delta_action);
                    s = ((obj.A*obj.dt + obj.I)*obj.state' + obj.B*obj.dt*action')';
                    obj.state = s;
                    new_state = [s, obj.ref_state(obj.counter+1,:), action];

            elseif obj.incremental == false && obj.S_A_dynamics == false
                    action = delta_action;
                    s = ((obj.A*obj.dt + obj.I)*obj.state' + obj.B*obj.dt*action')';
                    obj.state = s;
                    new_state = [s, obj.ref_state(obj.counter+1,:)];
            else
                error('Somthing is wrong with the environment settings concerning incremental actions and S&A dynamics')
            end
            internal_state = obj.state;    

            is_done = obj.is_done_function(s);
            reward = obj.reward_function(new_state, is_done);
            obj.is_done = is_done;
            obj.counter =  obj.counter + 1;
            %obj.last_action = action;
        end
        
        % Reset environment to initial state and output initial observation
        function state_0 = reset(obj)
            obj.counter = 1;
            % bounds for the reset values

            %create a vector with randomized values within the bounds
            x_reset = zeros(1, length(obj.state_vars));

            if obj.training == true
                obj.ref_state = zeros(obj.steps_per_episode+1, length(obj.state_vars));
                % for idx = obj.tracked_states
                %     if idx == obj.q || idx == obj.theta
                %         theta_max = 6; % degrees
                %         obj.ref_state(:,idx) = deg2rad(theta_max-2*theta_max*rand) * sin(0.6*rand*pi*obj.ep_length');
                %     end
                %     if idx == obj.p || idx == obj.phi
                %         phi_max = 4; % degrees
                %         obj.ref_state(:,idx) = deg2rad(phi_max -2*phi_max*rand) * sin(0.6*rand*pi*obj.ep_length');
                %     end                    
                % end
                idx = datasample(obj.tracked_idx,1);
                max_amp = 6; % degrees
                obj.ref_state(:,idx) = deg2rad(max_amp-2*max_amp*rand) * sin(0.6*rand*pi*obj.ep_length');
                obj.ref_state = obj.ref_state(:,obj.tracked_idx);

                for i = 1:length(obj.state_vars)
                    x_reset(i) = obj.reset_up_bound(i) + (obj.reset_lo_bound(i) - ...
                        obj.reset_up_bound(i)) * rand;
                end
                obj.ref_state = obj.ref_state + x_reset(obj.tracked_idx);
            end


            if obj.incremental == true
                state_0 = [x_reset, obj.ref_state(obj.counter,1:end), zeros(1, obj.bounds.action_space_size)];
            else
                state_0 = [x_reset, obj.ref_state(obj.counter,1:end)];
            end

            obj.state = x_reset;

            obj.last_action = zeros(1, obj.bounds.action_space_size);
            if obj.S_A_dynamics == true
                if any(nonzeros(obj.b_rates_idx))
                    obj.b_rate_buffer = x_reset(nonzeros(obj.b_rates_idx)) .* ones(sum(obj.b_rates_logical), round(obj.b_rate_delay / obj.dt))';
                end
                if any(nonzeros(obj.V_TAS_idx))
                    obj.V_TAS_buffer = x_reset(nonzeros(obj.V_TAS_idx)) .* ones(sum(obj.V_TAS_logical)), round(obj.V_TAS_delay / obj.dt)';
                end
                if any(nonzeros(obj.euler_idx))
                    obj.euler_buffer = x_reset(nonzeros(obj.euler_idx)) .* ones(sum(obj.euler_logical), round(obj.euler_delay / obj.dt))';
                end
                if any(nonzeros(obj.aero_idx))
                    obj.aero_buffer = x_reset(nonzeros(obj.aero_idx)) .* ones(sum(obj.aero_logical), round(obj.aero_delay / obj.dt))';
                end
                if any(nonzeros(obj.alt_idx))
                    obj.alt_buffer = x_reset(nonzeros(obj.alt_idx)) .* ones(sum(obj.alt_logical), round(obj.alt_delay / obj.dt))';
                end
                obj.x_servos = zeros(1,obj.bounds.action_space_size);
                obj.servo_buffer = zeros(round(obj.servo_delay / obj.dt), obj.bounds.action_space_size);
            end
            %obj.x_servos = obj.u0(1:obj.bounds.action_space_size);
        end
    
       function stop(obj)
            % Not used in this environment because its only needed when
            % using Simulink
        end

        function reward = reward_function(obj, new_state, is_done)
            % reward = -(180/pi)*sum(obj.error_cost.*(obj.ref_state(obj.counter+1,:) - new_state(obj.tracked_idx)).^2);
            if ~is_done
                reward = -(180/pi)*sum((obj.ref_state(obj.counter+1,:) - new_state(obj.tracked_idx)).^2);
            else
                reward = -10;
            end
            %reward = -sum(((180/pi)*(obj.ref_state(obj.counter+1,:) - new_state(obj.tracked_idx))).^2);
        end


        function is_done = is_done_function(obj, new_state)
            %check if limits have been exceeded
            up_lim_exceeded = new_state > obj.bounds.states_upper_bounds(1:length(obj.state_vars));
            lo_lim_exceeded = new_state < obj.bounds.states_lower_bounds(1:length(obj.state_vars));
            
            if any(up_lim_exceeded)
                is_done = true;
                var_id = obj.outvec(find(up_lim_exceeded),:);
                disp("Terminating because an upper limit is exceeded for state: " + var_id + new_state(up_lim_exceeded)')
            elseif any(lo_lim_exceeded)
                is_done = true;
                var_id = obj.outvec(find(lo_lim_exceeded),:);
                disp("Terminating because a lower limit is exceeded for state: " + var_id + new_state(lo_lim_exceeded)')
            else
                is_done = false;
                %disp('not done')
            end
        end

        function action = servo_dynamics(obj, action)
            if obj.incremental == true
                action = obj.last_action + action;
                up_lim_exceeded = action > obj.up_sat_lim;
                low_lim_exceeded = action < obj.lo_sat_lim;
                action(up_lim_exceeded) = obj.up_sat_lim(up_lim_exceeded);
                action(low_lim_exceeded) = obj.lo_sat_lim(low_lim_exceeded);
                obj.last_action = action;
            end
            if obj.S_A_dynamics == true
                obj.x_servos = ((obj.servos.A*obj.dt + 1).*obj.x_servos' + obj.servos.B*obj.dt.*action')';
                y_servos = obj.servos.C.*obj.x_servos' + obj.servos.D.*action';
                action = y_servos';

                obj.servo_buffer(end+1,:) = action;
                %action_mean = obj.servo_buffer(1,:); % with delay
                action_mean = obj.servo_buffer(end,:); % without delay
                action = obj.servo_bias + normrnd(action_mean, obj.servo_std, [1, obj.bounds.action_space_size]);
                obj.servo_buffer(1,:) = [];
            end



        end


        function noisy_state = sensor_dynamics(obj, new_state)
        
            noisy_state = new_state;

            if any(obj.b_rates_logical)
                %disp('b rates')
                obj.b_rate_buffer(end+1,:) = new_state(nonzeros(obj.b_rates_idx));
                %b_rate_mean = obj.b_rate_buffer(1,:); % with delay
                b_rate_mean = obj.b_rate_buffer(end,:); % without delay
                noisy_state(nonzeros(obj.b_rates_idx)) = obj.b_rate_bias + normrnd(b_rate_mean, obj.b_rate_std, [1, length(nonzeros(obj.b_rates_idx))]);
                obj.b_rate_buffer(1,:) = [];
            end
            if any(obj.V_TAS_logical)
                %disp('V_TAS')
                obj.V_TAS_buffer(end+1,:) = new_state(nonzeros(obj.V_TAS_idx));
                V_TAS_mean = obj.V_TAS_buffer(1,:); % with delay
                noisy_state(nonzeros(obj.V_TAS_idx)) = obj.V_TAS_bias + normrnd(V_TAS_mean, obj.V_TAS_std, [1, length(nonzeros(obj.V_TAS_idx))]);
                obj.V_TAS_buffer(1,:) = [];
            end
            if any(obj.euler_logical)
                %disp('eurler')
                obj.euler_buffer(end+1,:) =  new_state(nonzeros(obj.euler_idx));
                %euler_mean = obj.euler_buffer(1,:); % with delay
                euler_mean = obj.euler_buffer(end,:); % without delay
                noisy_state(nonzeros(obj.euler_idx)) = obj.euler_bias + normrnd(euler_mean, obj.euler_std, [1, length(nonzeros(obj.euler_idx))]);
                obj.euler_buffer(1,:) = [];
            end
            if any(obj.aero_logical)
                %disp('aero')
                obj.aero_buffer(end+1,:) = new_state(nonzeros(obj.aero_idx));
                %aero_mean = obj.aero_buffer(1,:); % with delay
                aero_mean = obj.aero_buffer(end,:); % without delay
                noisy_state(nonzeros(obj.aero_idx)) = obj.aero_bias + normrnd(aero_mean, obj.aero_std, [1, length(nonzeros(obj.aero_idx))]);
                obj.aero_buffer(1,:) = [];
            end
            if any(obj.alt_logical)
                %disp('alt')
                obj.alt_buffer(end+1,:) = new_state(nonzeros(obj.alt_idx));
                alt_mean = obj.alt_buffer(1,:); % with delay
                noisy_state(nonzeros(obj.alt_idx)) = obj.alt_bias + normrnd(alt_mean, obj.alt_std, [1, length(nonzeros(obj.alt_idx))]);
                obj.alt_buffer(1,:) = [];
            end
        end
    end
end
