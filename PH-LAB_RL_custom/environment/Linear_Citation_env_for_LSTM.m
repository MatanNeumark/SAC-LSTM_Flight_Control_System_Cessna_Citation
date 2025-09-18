classdef Linear_Citation_env_for_LSTM < rl.env.MATLABEnvironment
    properties
        state_vars_dict = dictionary(1, "p", 2, "q", 3, "r", 4, "V_TAS", 5, "alpha", 6, "beta", 7, "phi", 8, "theta", 9,  "psi", 10, "he", 11, "xe", 12, "ye")
        tracked_states_dict = dictionary(1, "p_{ref}", 2, "q_{ref}", 3, "r_{ref}", 4, "V_TAS_{ref}", 5, "alpha_{ref}", 6, "beta_{ref}", 7, "phi_{ref}", 8, "theta_{ref}", 9,  "psi_{ref}", 10, "he_{ref}", 11, "xe_{ref}", 12, "ye_{ref}")
        %input_dict = dictionary(1, 'elevator', 2, 'aileron', 3, 'rudder')
        p = 1; % roll rate
        q = 2; % pitch rate
        r = 3; % yaw rate
        V_TAS = 4; % true airspeed
        alpha = 5; % angle of attack
        beta = 6; % sideslip angle
        phi = 7; % bank angle
        theta = 8; % pitch angle (euler)
        psi = 9; % heading angle (euler)
        h = 10; % altitude
        ref_state % the reference tracking signal
        dt = 0.01 % time step increment (seconds)
        steps_per_episode % number of time steps per episode
        state_vars % vector of state variables
        tracked_states % the tracked state variables
        SA_noise_bias % sensors and servos noise and bias. 'False' is recommended for training
        SA_delay % sensors and servos delay. 'False' is recommended for training
        incremental % incremental control
        training % if 'true', the reference signal frequency and amplitude is chosen at random every episode, otherwise it uses the supplied ref_state. "true" also enables exploring starts.
        env_settings % logging of the environment settings
        tracked_idx % the indices of the tracked state variables
        ep_length % episode length in seconds
        bounds % environment boundaries
        reset_up_bound % vector with the values of the upper bounds
        reset_lo_bound % vector with the values of the lower bounds
        lin_sys % state space system of the linearised dynamics of the aircraft
        I % identity matrix
        x0 % trim conditions. (loaded to workspace when loading the trim file)
        up_sat_lim % upper servo saturation limits
        lo_sat_lim % lower servo saturation limits
        outvec % string vector of state variables from the workspace (loaded to workspace when loading the trim file)
        de_ss % state space system for the elevator
        da_ss % state space system for the ailerons
        dr_ss % state space system for the rudder
        servo_RL % servo rate limit. 'False' is recommended for training because it hinders exploration
        servo_TF % First order transfer function to model the servo. 'True' is recommended for training so the agent can learn these dynamics
        servos % a struct to store the state space systems of all the servos
        x_servos % servo state vector
        u0 % initial control surface deflections. (loaded to workspace when loading the trim file)
        state % the current state of the aircraft
        last_action % the last commanded action
        is_terminated = false % episode termination flag
        counter = 1 % time step counter
        error_cost % can be used to balance the cost of the various tracking errors
        % de_up_sat_lim = deg2rad(15); % upper limit
        % de_lo_sat_lim = deg2rad(-17); % lower limit
        % da_sat_lim = deg2rad(22); % symmetric
        % dr_sat_lim = deg2rad(34); % symmetric
        % de_rate_lim = deg2rad(20) % limit is 20 deg/s
        % da_rate_lim = deg2rad(20) % limit is 20 deg/s
        % dr_rate_lim = deg2rad(40) % limit is 40 deg/s
        b_rates_logical
        b_rates_idx % indices of the body rates in the state vector
        V_TAS_logical
        V_TAS_idx % index of the true airspeed in the state vector
        euler_logical
        euler_idx % indices of the euler angles in the state vector
        aero_logical
        aero_idx % indices of the aerodynamic angles in the state vector
        alt_logical
        alt_idx % index of the altitude in the state vector

        b_rate_bias %= 3e-5;
        b_rate_std %= sqrt(4e-7);
        b_rate_delay %= 15e-3;
        b_rate_freq = 1000; % sample frequency [Hz]
        b_rate_buffer % buffer is used when delay in enabled

        V_TAS_bias %= 2.5;
        V_TAS_std %= sqrt(8.5e-4);
        V_TAS_delay %= 300e-3;
        V_TAS_freq = 8; % sample frequency [Hz]
        V_TAS_buffer % buffer is used when delay in enabled

        euler_bias %= 4e-3;
        euler_std %= sqrt(1e-9);
        euler_delay %= 90e-3;
        euler_freq = 52; % sample frequency [Hz]
        euler_buffer % buffer is used when delay in enabled

        aero_bias %= 1.8e-3;
        aero_std %= sqrt(7.5e-8);
        aero_delay %= 100e-3;
        aero_freq = 100; % sample frequency [Hz]
        aero_buffer % buffer is used when delay in enabled

        alt_bias % = 8e-3;
        alt_std %= sqrt(4.5e-3);
        alt_delay %= 300e-3;
        alt_freq = 20; % sample frequency [Hz]
        alt_buffer % buffer is used when delay in enabled

        servo_bias %= 2.4e-3;
        servo_std %= sqrt(5.5e-7);
        servo_delay %= 10e-3;
        servo_freq = 100; % sample frequency [Hz]
        servo_buffer % buffer is used when delay in enabled
    end
    

    methods              
        function obj = Linear_Citation_env_for_LSTM(ref_state, dt, steps_per_episode, state_vars, tracked_states, SA_noise_bias, SA_delay, incremental, training, servo_RL, servo_TF)
            env_config; % loads the environment configuration file. It defines the servo saturation limits and loads the trim file
            delta_de_lim = de_rate_lim * dt; % elevator deflection limitation per time step when considering the rate limit
            delta_da_lim = da_rate_lim * dt; % ailerons deflection limitation per time step when considering the rate limit
            delta_dr_lim = dr_rate_lim * dt; % rudder deflection limitation per time step when considering the rate limit
            % ObservationInfo = rlNumericSpec([1 12]);
            if width(ref_state) > height(ref_state) % arrange the reference state vector in the right format
                ref_state = ref_state';
            end
            if all(~ismember([p, r, phi, Beta], state_vars)) && min(size(ref_state)) == 1 && incremental == true % if longitudinal and incremental control
                %ActionInfo = rlNumericSpec([1 1], LowerLimit=-delta_de_lim, UpperLimit=delta_de_lim);
                ActionInfo = rlNumericSpec([1 1], LowerLimit=de_lo_sat_lim, UpperLimit=de_up_sat_lim); % elevator only
                ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states) + 2 * ActionInfo.Dimension(2)]); % state vector includes the state variables, the reference signals for the tracked state variables, the commanded deflections, and the applied deflections 
                ActionInfo.Name = 'Elevator';           
            elseif all(~ismember([p, r, phi, Beta], state_vars)) && min(size(ref_state)) == 1 && incremental == false % if longitudinal and not incremental control
                ActionInfo = rlNumericSpec([1 1], LowerLimit=de_lo_sat_lim, UpperLimit=de_up_sat_lim); % elevator only
                % ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states)]);
                ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states) + ActionInfo.Dimension(2)]); % state vector includes the state variables, the reference signals for the tracked state variables, the commanded deflections, and the applied deflections 
                ActionInfo.Name = 'Elevator';           
            elseif all(ismember([p, q, r, phi, Beta], state_vars)) && min(size(ref_state)) == 3 && incremental == true % if full DoF (longitudinal and lateral) and incremental control
                %ActionInfo = rlNumericSpec([1 3], LowerLimit=[-delta_de_lim, -delta_da_lim, -delta_dr_lim], UpperLimit=[delta_de_lim, delta_da_lim, delta_dr_lim]);
                ActionInfo = rlNumericSpec([1 3], LowerLimit=[de_lo_sat_lim, -da_sat_lim, -dr_sat_lim], UpperLimit=[de_up_sat_lim, da_sat_lim, dr_sat_lim]); % elevator, ailerons, rudder
                ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states) + 2 * ActionInfo.Dimension(2)]); % state vector includes the state variables, the reference signals for the tracked state variables, the commanded deflections, and the applied deflections 
                ActionInfo.Name = 'Elevator, Aileron, Rudder';           
            elseif all(ismember([p, q, r, phi, Beta], state_vars)) && min(size(ref_state)) == 3 && incremental == false % if full DoF (longitudinal and lateral) and not incremental control
                ActionInfo = rlNumericSpec([1 3], LowerLimit=[de_lo_sat_lim, -da_sat_lim, -dr_sat_lim], UpperLimit=[de_up_sat_lim, da_sat_lim, dr_sat_lim]); % elevator, ailerons, rudder
                ObservationInfo = rlNumericSpec([1 length(state_vars) + length(tracked_states) +  ActionInfo.Dimension(2)]); % state vector includes the state variables, the reference signals for the tracked state variables, the commanded deflections, and the applied deflections 
                ActionInfo.Name = 'Elevator, Aileron, Rudder';           
            else
                error("Please define a valid set of state variables and reference signals")
            end

            ObservationInfo.Name = 'Citation States';
            ObservationInfo.Description = num2str(state_vars);
            % The following line implements built-in functions of RL env
            obj = obj@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);

            obj.ref_state = ref_state;            
            obj.dt = dt; % time step increment
            obj.steps_per_episode = steps_per_episode;
            obj.state_vars = state_vars;
            obj.tracked_states = tracked_states;
            obj.SA_noise_bias = SA_noise_bias;  % true or false
            obj.SA_delay = SA_delay;  % true or false
            obj.incremental = incremental; % true or false
            obj.training =training;  % true or false
            obj.servo_RL = servo_RL;  % true or false
            obj.servo_TF = servo_TF;  % true or false
            obj.env_settings = dictionary(1, ['SA_noise_bias ', num2str(SA_noise_bias)],...
                                          2, ['SA_delay ', num2str(SA_delay)],...
                                          3, ['incremental ', num2str(incremental)],...
                                          4, ['training ', num2str(training)],...
                                          5, ['servo_RL ', num2str(servo_RL)],...
                                          6, ['servo_TF ', num2str(servo_TF)]);

            obj.bounds.state_space_size = ObservationInfo.Dimension(2); % size of the state space
            obj.bounds.action_space_size = ActionInfo.Dimension(2); % size of the action space
            obj.bounds.n_ref_states = length(tracked_states); % number of tracked state variables
            obj.bounds.ref_states_idx = length(state_vars) + 1:length(state_vars) + obj.bounds.n_ref_states; % indices of the reference signals in the state vector

            %load Matan_v90ms_m4500kg_h2000m_q_init.lin -mat
            load('environment/full_DoF_v90ms_m4500kg_h2000m_q_init.lin', '-mat') % load the desired linearised system

            obj.up_sat_lim = [de_up_sat_lim, da_sat_lim, dr_sat_lim]; % upper saturation limits of all servos
            obj.up_sat_lim = obj.up_sat_lim(1:obj.bounds.action_space_size); % truncation according to the size of the action space
            obj.lo_sat_lim = [de_lo_sat_lim, -da_sat_lim, -dr_sat_lim]; % lower saturation limits of all servos
            obj.lo_sat_lim = obj.lo_sat_lim(1:obj.bounds.action_space_size); % truncation according to the size of the action space
            %obj.delta_lim = [delta_de_lim, delta_da_lim, delta_dr_lim];
            %obj.delta_lim = obj.delta_lim(1:obj.bounds.action_space_size);
            obj.servos.rate_lim = [de_rate_lim, da_rate_lim, dr_rate_lim]; % rate limit of all servos
            obj.servos.rate_lim = obj.servos.rate_lim(1:obj.bounds.action_space_size); % truncation according to the size of the action space
            obj.servos.tau = 1/13; % time constant of the servos. used in a first order transfer function

            obj.u0 = u0;
            [~, obj.tracked_idx] = ismember(obj.tracked_states, obj.state_vars);
            obj.outvec = outvec(obj.state_vars,:);

            A = Alin(obj.state_vars, obj.state_vars); % the A matrix of the linear system of the aircraft
            B = Blin(obj.state_vars, 1:ActionInfo.Dimension(2)); % the B matrix of the linear system of the aircraft
            C = Clin(obj.state_vars, obj.state_vars); % the C matrix of the linear system of the aircraft
            D = Dlin(obj.state_vars, 1:ActionInfo.Dimension(2)); % the D matrix of the linear system of the aircraft
            obj.lin_sys = ss(A, B, C, D,...
                    'statename', obj.state_vars_dict(obj.state_vars), ...
                    'inputname', ActionInfo.Name,...
                    'outputname', obj.state_vars_dict(obj.state_vars));

            obj.x0 = x0;
            obj.x0([obj.alpha, obj.beta, obj.theta]) = 0;
            obj.x0 = obj.x0(obj.state_vars);

            obj.de_ss = ss(tf(13, [1 13])); % elevator first order transfer function in state space form
            obj.da_ss = ss(tf(13, [1 13])); % ailerons first order transfer function in state space form
            obj.dr_ss = ss(tf(13, [1 13])); % rudder first order transfer function in state space form

            obj.servos.A = [obj.de_ss.A; obj.da_ss.A; obj.dr_ss.A];
            obj.servos.A = obj.servos.A(1:obj.bounds.action_space_size); 
            obj.servos.B = [obj.de_ss.B; obj.da_ss.B; obj.dr_ss.B];
            obj.servos.B = obj.servos.B(1:obj.bounds.action_space_size); 
            obj.servos.C = [obj.de_ss.C; obj.da_ss.C; obj.dr_ss.C];
            obj.servos.C = obj.servos.C(1:obj.bounds.action_space_size); 
            obj.servos.D = [obj.de_ss.D; obj.da_ss.D; obj.dr_ss.D];
            obj.servos.D = obj.servos.D(1:obj.bounds.action_space_size); 

            obj.bounds.states_upper_bounds = [deg2rad(20), deg2rad(10), deg2rad(10), 130, deg2rad(10), deg2rad(10), deg2rad(20), deg2rad(10), inf, 2300, inf, inf]; % upper bounds for the variables in the state vector
            obj.bounds.states_lower_bounds = [-deg2rad(20), -deg2rad(10), -deg2rad(10), 50, -deg2rad(10), -deg2rad(10), -deg2rad(20), -deg2rad(10), -inf, 1700, -inf, -inf]; % lower bounds for the variables in the state vector
            if incremental == true
                obj.bounds.states_upper_bounds = [obj.bounds.states_upper_bounds(obj.state_vars), obj.bounds.states_upper_bounds(obj.tracked_states), obj.up_sat_lim(1:ActionInfo.Dimension(2)), obj.up_sat_lim(1:ActionInfo.Dimension(2))];
                obj.bounds.states_lower_bounds = [obj.bounds.states_lower_bounds(obj.state_vars), obj.bounds.states_lower_bounds(obj.tracked_states), obj.lo_sat_lim(1:ActionInfo.Dimension(2)), obj.lo_sat_lim(1:ActionInfo.Dimension(2))];
            else
                obj.bounds.states_upper_bounds = [obj.bounds.states_upper_bounds(obj.state_vars), obj.bounds.states_upper_bounds(obj.tracked_states), obj.up_sat_lim(1:ActionInfo.Dimension(2))];
                obj.bounds.states_lower_bounds = [obj.bounds.states_lower_bounds(obj.state_vars), obj.bounds.states_lower_bounds(obj.tracked_states), obj.lo_sat_lim(1:ActionInfo.Dimension(2))];
            end

            obj.bounds.actions_upper_bound = ActionInfo.UpperLimit; % upper bounds for the actions according to the saturation limits
            obj.bounds.actions_lower_bound = ActionInfo.LowerLimit; % upper bounds for the actions according to the saturation limits
            obj.reset_up_bound = 1e-2 * obj.bounds.states_upper_bounds; % upper bounds for exploring starts
            obj.reset_lo_bound = 1e-2 * obj.bounds.states_lower_bounds; % lower bounds for exploring starts
            obj.I = eye(length(obj.state_vars));

            obj.error_cost = [1, 1, 1, 1, 1, 4, 1, 1, 1, 1]; % can be used to balance the cost of the various tracking errors
            obj.error_cost = obj.error_cost(obj.tracked_states);
            obj.error_cost = obj.error_cost/sum(obj.error_cost);

            [obj.b_rates_logical, obj.b_rates_idx] = ismember([obj.p, obj.q, obj.r], obj.state_vars); % checks which body rates are part of the state vector
            [obj.V_TAS_logical, obj.V_TAS_idx] = ismember(obj.V_TAS, obj.state_vars); % checks if the true airspeed is part of the state vector
            [obj.euler_logical, obj.euler_idx] = ismember([obj.phi, obj.theta, obj.psi], obj.state_vars); % checks which euler angles are part of the state vector
            [obj.aero_logical, obj.aero_idx] = ismember([obj.alpha, obj.beta], obj.state_vars); % checks which aerodynamics angles are part of the state vector
            [obj.alt_logical, obj.alt_idx] = ismember(obj.h, obj.state_vars); % checks if altitude is part of the state vector



        end
        

        function [new_state, reward, terminated, Info, applied_action, internal_state] = step(obj, action)
            % This function simulates the aircraft dynamics on a time step basis
            Info = []; % needed when using MATLAB's RL toolbox with this environment

            if any(isnan(action))
                error("The action is NaN")
            end

            applied_action = obj.servo_dynamics(action)'; % servo dynamics
            s = ((obj.lin_sys.A*obj.dt + obj.I)*obj.state' + obj.lin_sys.B*obj.dt*applied_action)'; % compute the new state of the aircraft
            obj.state = s;
            s = obj.sensor_dynamics(s); % sensor dynamics
            % 
            if obj.incremental
                new_state = [s, obj.ref_state(obj.counter+1,:), applied_action', obj.last_action]; % if incremental, add the last action to the state vector
            else
                new_state = [s, obj.ref_state(obj.counter+1,:), applied_action'];
            end

            internal_state = obj.state;
            terminated = obj.termination_function(s); % check if agent exceeds the environment bounds
            reward = obj.reward_function(s, action, terminated); % calculate the reward

            obj.is_terminated = terminated; % save the termination status
            obj.counter =  obj.counter + 1; % increment the time step counter
        end

        
        function [state_0, action_0] = reset(obj)
            % Reset the environment to initial state and output the initial observation
            obj.counter = 1; % reset the time step counter

            if obj.SA_noise_bias % noise and bias values if enabled
                obj.b_rate_bias = 3e-5;
                obj.b_rate_std = sqrt(4e-7);
                obj.V_TAS_bias = 2.5;
                obj.V_TAS_std = sqrt(8.5e-4);        
                obj.euler_bias = 4e-3;
                obj.euler_std = sqrt(1e-9);       
                obj.aero_bias = 1.8e-3;
                obj.aero_std = sqrt(7.5e-8);
                obj.alt_bias = 8e-3;
                obj.alt_std = sqrt(4.5e-3);        
                obj.servo_bias = 2.4e-3;
                obj.servo_std = sqrt(5.5e-7);
            else
                obj.b_rate_bias = 0;
                obj.b_rate_std = 0;        
                obj.V_TAS_bias = 0;
                obj.V_TAS_std = 0;
                obj.euler_bias = 0;
                obj.euler_std = 0;
                obj.aero_bias = 0;
                obj.aero_std = 0;        
                obj.alt_bias = 0;
                obj.alt_std = 0;      
                obj.servo_bias = 0;
                obj.servo_std = 0;
            end
            if obj.SA_delay % delay values if enables (in milliseconds)
                obj.b_rate_delay = 15e-3;
                obj.V_TAS_delay = 300e-3;
                obj.euler_delay = 90e-3;
                obj.aero_delay = 100e-3;
                obj.alt_delay = 300e-3;
                obj.servo_delay = 10e-3;
            else
                obj.b_rate_delay = 0;
                obj.V_TAS_delay = 0;
                obj.euler_delay = 0;
                obj.aero_delay = 0;
                obj.alt_delay = 0;
                obj.servo_delay = 0;
            end
            %create a vector with randomized values within the bounds (exploring starts)
            x_reset = zeros(1, length(obj.state_vars)); % initialise the initial state vector
            obj.ep_length = 0:obj.dt:obj.dt*obj.steps_per_episode;
            if obj.training == true
                obj.ref_state = zeros(obj.steps_per_episode+1, length(obj.state_vars));
                idx = datasample(obj.tracked_idx,1);
                min_amp = deg2rad(datasample([1, -1], 1)); % degrees
                max_amp = deg2rad(4); % degrees
                % ref = datasample(["step", "sine"], 1);
                % if ref == "step"
                %     step_time = randsample(obj.steps_per_episode/2,1);
                %     obj.ref_state(:,idx) = (min_amp + (max_amp-2*max_amp*rand)) * [zeros(1,step_time), ones(1,obj.steps_per_episode - step_time + 1)];
                %     obj.ref_state = obj.ref_state(:,obj.tracked_idx);
                % else
                    max_freq = 0.2;%0.4;
                    obj.ref_state(:,idx) = (min_amp + (max_amp-2*max_amp*rand)) * sin(max_freq*rand*pi*obj.ep_length'); % creates sine wave reference signal with a random amplitude and frequency according the the limits above.
                    obj.ref_state = obj.ref_state(:,obj.tracked_idx);
                %end
                for i = 1:length(obj.state_vars)
                    x_reset(i) = obj.reset_up_bound(i) + (obj.reset_lo_bound(i) - ...
                                 obj.reset_up_bound(i)) * rand; % exploring starts
                end
                obj.ref_state = obj.ref_state + x_reset(obj.tracked_idx);
            end

            obj.state = x_reset;

            obj.x_servos = zeros(1,obj.bounds.action_space_size); % initialises the control surface deflections to zero
            action_0 = obj.x_servos;
            obj.last_action = obj.x_servos;
            if obj.incremental == true
                state_0 = [x_reset, obj.ref_state(obj.counter,1:end), obj.x_servos, obj.x_servos];
            else
                state_0 = [x_reset, obj.ref_state(obj.counter,1:end), obj.x_servos];
            end
            
            % creates buffers for the state variables according to the length of delay they have (in time steps)
            if any(nonzeros(obj.b_rates_idx))
                obj.b_rate_buffer = x_reset(nonzeros(obj.b_rates_idx)) .* ones(sum(obj.b_rates_logical), round(obj.b_rate_delay / obj.dt))';
            end
            if any(nonzeros(obj.V_TAS_idx))
                obj.V_TAS_buffer = x_reset(nonzeros(obj.V_TAS_idx)) .* ones(sum(obj.V_TAS_logical), round(obj.V_TAS_delay / obj.dt))';
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
            obj.servo_buffer = zeros(round(obj.servo_delay / obj.dt), obj.bounds.action_space_size);
        end
        
       function stop(obj)
           disp("This function doesn't do anything in this environment")
            % Not used in this environment because its only needed when using Simulink
        end

        function reward = reward_function(obj, new_state, action, terminated)
            if ~terminated %|| (terminated && ismember(obj.tracked_states, obj.state_vars(obj.lim_exceeded)))
                reward = -(180/pi)*sum((obj.ref_state(obj.counter+1,:) - new_state(obj.tracked_idx)).^2); % scaled, squared tracking error
                reward = clip(reward, -1, 0); % clips the reward to make it 'normalised'
            else
                reward = -1; % penalty if the agent terminates the episode
            end            
        end


        function terminated = termination_function(obj, new_state)
            %check if limits have been exceeded
            up_lim_exceeded = new_state > obj.bounds.states_upper_bounds(1:length(obj.state_vars)); % check if any state variable exceeds the upper bounds
            lo_lim_exceeded = new_state < obj.bounds.states_lower_bounds(1:length(obj.state_vars)); % check if any state variable exceeds the lower bounds
            lim_exceeded = up_lim_exceeded | lo_lim_exceeded;
            
            if any(lim_exceeded)
              terminated = true;
              var_id = obj.outvec(find(lim_exceeded),:);
              disp("Terminating because a limit is exceeded for state: " + var_id + new_state(lim_exceeded)') % print which state variable was exceeded, and its value
            else
              terminated = false;
            end
        end

        function action = servo_dynamics(obj, action)
            if obj.incremental
                action = obj.last_action + action - obj.servo_bias; % to prevent position drift with incremental control, the bias is subtract here
                up_lim_exceeded = action > obj.up_sat_lim;
                low_lim_exceeded = action < obj.lo_sat_lim;
                action(up_lim_exceeded) = obj.up_sat_lim(up_lim_exceeded);
                action(low_lim_exceeded) = obj.lo_sat_lim(low_lim_exceeded);
                obj.last_action = action; % store the last position
            end

            if obj.servo_TF % applies the affect of a first order transfer function
                obj.x_servos = ((obj.servos.A*obj.dt + 1).*obj.x_servos' + obj.servos.B*obj.dt.*action')';
                y_servos = obj.servos.C.*obj.x_servos' + obj.servos.D.*action';
                action = y_servos';
            end

            if obj.servo_RL % applies the affect of a rate limited servo
                action = clip(action, obj.last_action-obj.servos.rate_lim*obj.dt, obj.last_action+obj.servos.rate_lim*obj.dt); % comment out if using incremental;
                action = clip(action, obj.lo_sat_lim, obj.up_sat_lim);
                obj.last_action = action;
            end

            obj.servo_buffer(end+1,:) = action; % append the action to the end of the buffer
            action_mean = obj.servo_buffer(1,:); % sample the first action in the buffer
            action = action_mean + obj.servo_bias + obj.servo_std*randn([1, obj.bounds.action_space_size]); % add noise and bias
            obj.servo_buffer(1,:) = []; % remove the first sample from the buffer to preserve its length

            % if obj.S_A_dynamics % alternative model for the servo dynamics
                % action_error = action - obj.last_action;
                % rate = action_error / obj.servos.tau
                % sat_rate = clip(rate, -obj.servos.rate_lim, obj.servos.rate_lim)
                % action = obj.last_action + sat_rate*obj.dt;
                % action = clip(action, obj.lo_sat_lim, obj.up_sat_lim);
                % obj.last_action = action;

                % obj.servo_buffer(end+1,:) = action;
                % action_mean = obj.servo_buffer(1,:); % with delay
                % %action_mean = obj.servo_buffer(end,:); % without delay
                % %action = obj.servo_bias + normrnd(action_mean, obj.servo_std, [1, obj.bounds.action_space_size]);
                % action = action_mean + obj.servo_bias + obj.servo_std*randn([1, obj.bounds.action_space_size]);
                % %action = action_mean;
                % obj.servo_buffer(1,:) = [];

            % end
        end


        function noisy_state = sensor_dynamics(obj, new_state)
        
            noisy_state = new_state;

            if any(obj.b_rates_logical)
                obj.b_rate_buffer(end+1,:) = new_state(nonzeros(obj.b_rates_idx)); % append the state variable to the end of the buffer
                b_rate_mean = obj.b_rate_buffer(1,:); % sample the first values in the buffer
                noisy_state(nonzeros(obj.b_rates_idx)) = b_rate_mean + obj.b_rate_bias + obj.b_rate_std*randn([1, length(nonzeros(obj.b_rates_idx))]); % add noise and bias
                obj.b_rate_buffer(1,:) = []; % remove the first sample from the buffer to preserve its length
            end
            if any(obj.V_TAS_logical)
                obj.V_TAS_buffer(end+1,:) = new_state(nonzeros(obj.V_TAS_idx)); % append the state variable to the end of the buffer
                V_TAS_mean = obj.V_TAS_buffer(1,:);  % sample the first values in the buffer
                noisy_state(nonzeros(obj.V_TAS_idx)) = V_TAS_mean + obj.V_TAS_bias + obj.V_TAS_std*randn([1, length(nonzeros(obj.V_TAS_idx))]); % add noise and bias
                obj.V_TAS_buffer(1,:) = []; % remove the first sample from the buffer to preserve its length
            end
            if any(obj.euler_logical)
                obj.euler_buffer(end+1,:) =  new_state(nonzeros(obj.euler_idx)); % append the state variable to the end of the buffer
                euler_mean = obj.euler_buffer(1,:);  % sample the first values in the buffer
                noisy_state(nonzeros(obj.euler_idx)) = euler_mean + obj.euler_bias + obj.euler_std*randn([1, length(nonzeros(obj.euler_idx))]); % add noise and bias
                obj.euler_buffer(1,:) = []; % remove the first sample from the buffer to preserve its length
            end
            if any(obj.aero_logical)
                obj.aero_buffer(end+1,:) = new_state(nonzeros(obj.aero_idx)); % append the state variable to the end of the buffer
                aero_mean = obj.aero_buffer(1,:);  % sample the first values in the buffer
                noisy_state(nonzeros(obj.aero_idx)) = aero_mean + obj.aero_bias + obj.aero_std*randn([1, length(nonzeros(obj.aero_idx))]); % add noise and bias
                obj.aero_buffer(1,:) = []; % remove the first sample from the buffer to preserve its length
            end
            if any(obj.alt_logical)
                obj.alt_buffer(end+1,:) = new_state(nonzeros(obj.alt_idx)); % append the state variable to the end of the buffer
                alt_mean = obj.alt_buffer(1,:);  % sample the first values in the buffer
                noisy_state(nonzeros(obj.alt_idx)) = alt_mean + obj.alt_bias + obj.alt_std*randn([1, length(nonzeros(obj.alt_idx))]); % add noise and bias
                obj.alt_buffer(1,:) = []; % remove the first sample from the buffer to preserve its length
            end
        end
    end
end
