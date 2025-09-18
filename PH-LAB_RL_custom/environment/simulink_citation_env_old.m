classdef simulink_citation_env_old < handle
    properties
        mdl
        env_settings
        dt
        steps_per_episode
        episode_length
        de_sat_limit = 0.5;
        da_sat_limit = 0.5;
        dr_sat_limit = 0.5;
        trimdatafile = 'citast.tri';
        elevator_input
        aileron_input
        rudder_input
        state_to_use
        sm
        bounds
        ref_state
        counter = 1
        %reset_block_path
        %x_reset
        x0
    end

    methods
        function obj = simulink_citation_env(mdl, env_settings)
            Env_config;
            %obj.trimdatafile = trimdatafile;
            load(obj.trimdatafile,'-mat');
            obj.mdl = mdl;
            obj.env_settings = env_settings;
            obj.dt = env_settings('dt');
            obj.steps_per_episode = env_settings('steps_per_episode');
            obj.ref_state = obj.env_settings('ref_state');
            obj.episode_length = obj.steps_per_episode * obj.dt;
            assignin('base','Ts',obj.dt);
            obj.state_to_use = [5, 2, 2];
            obj.bounds.state_space_size = 3; %p ,q, r, vtas, alpha, beta, phi, theta, psi, he, xe, ye
            obj.bounds.action_space_size = 1; %de, da, dr
            obj.bounds.states_upper_bounds = [deg2rad(20), deg2rad(20), deg2rad(20), 200, deg2rad(20), deg2rad(20), deg2rad(20), deg2rad(20), deg2rad(20), 1e4, 1e5, 1e5];
            obj.bounds.states_upper_bounds = obj.bounds.states_upper_bounds(obj.state_to_use);
            obj.bounds.states_lower_bounds = [-deg2rad(20), -deg2rad(20), -deg2rad(20), 0, -deg2rad(20), -deg2rad(20), -deg2rad(20), -deg2rad(20), -deg2rad(20), 0, -1e5, -1e5];
            obj.bounds.states_lower_bounds = obj.bounds.states_lower_bounds(obj.state_to_use);
            obj.bounds.actions_upper_bound = [obj.de_sat_limit, obj.da_sat_limit, obj.dr_sat_limit];
            obj.bounds.actions_lower_bound = [-obj.de_sat_limit, -obj.da_sat_limit, -obj.dr_sat_limit];
            obj.x0 = x0;
            obj.elevator_input = strcat(obj.mdl,"/elevator input");
            obj.aileron_input = strcat(obj.mdl,"/aileron input");
            obj.rudder_input = strcat(obj.mdl,"/rudder input");
            %open_system(mdl);
            obj.sm = simulation(mdl);
            obj.sm = setModelParameter(obj.sm,StopTime=num2str(obj.episode_length + obj.dt));
            set_param(strcat(obj.mdl,...
                "/Cessna Citation 500 Non-Lineair 6 DOF model"), ...
                'trimdatafile', obj.trimdatafile);
            obj.sm.Status  
        end

        function [new_state, reward, is_done] = step(obj, action)
            if obj.sm.Status == "inactive"
               state_0 = obj.reset;
            end
            % make a step in the environment and return the new state
            % pass the action onto the Simulink model 
            if obj.env_settings('longitudinal') == true
                obj.sm = setBlockParameter(obj.sm, obj.elevator_input, "value" ,num2str(action(1)));
                %set_param(obj.elevator_input, "value" ,num2str(action(1)));
                obj.sm = setBlockParameter(obj.sm, obj.aileron_input, "value" ,num2str(0));
                obj.sm = setBlockParameter(obj.sm, obj.rudder_input, "value" ,num2str(0));
            else
                obj.sm = setBlockParameter(obj.sm, obj.elevator_input, "value" ,num2str(action(1)));
                obj.sm = setBlockParameter(obj.sm, obj.aileron_input, "value" ,num2str(action(2)));
                obj.sm = setBlockParameter(obj.sm, obj.rudder_input, "value" ,num2str(action(3)));
            end
            % advance the environment by one time step step_size
            step(obj.sm, PauseTime= obj.sm.Time + obj.dt);
            %param = get_param('Citation_RL_custom_env_by_Matan/elevator input', 'Value')
            % check new state
            new_state = obj.sm.SimulationOutput.xout.signals(1).values(end,:);
            %obj.sm.SimulationOutput.yout{1}.Values.Time(end)
            % check if the new state is terminal
            is_done = obj.is_done_function(new_state);
            % determine reward
            reward = obj.reward_function(obj.env_settings('ref_state'), new_state);
            reward_blk = strcat(obj.mdl,"/reward block");
            obj.sm = setBlockParameter(obj.sm, reward_blk, "value" ,num2str(reward));
            
            %pause(8)
            obj.ref_state(obj.counter+1)
            if obj.env_settings('longitudinal') == true
                %new_state = new_state([2, 4, 5, 8, 10]); # q, vtas, alpha, theta, he
                new_state = [new_state([5, 2]), obj.ref_state(obj.counter+1)]; % [alpha, q]
            end
            obj.counter = obj.counter + 1;

        end


        function state_0 = reset(obj)
            obj.counter = 1;
            % check environment status and stop only if it is
            % "initialized", "running" or "paused"
            if ismember(obj.sm.Status, ["initialized", "running", "paused"])
                stop(obj.sm);
            end

            % bounds for the reset values
            if obj.env_settings('longitudinal') == true
                %                 p ,q,          r,  vtas, alpha, beta, phi, theta,     psi, he, xe, ye
                reset_up_bound = [0, deg2rad(0), 0,  0,    0,     0,    0, deg2rad(0),  0,   0, 0,  0];
                reset_lo_bound = [0, deg2rad(0), 0,  0,    0,     0,    0, deg2rad(0),  0,   0, 0,  0];
            else
                %                 p,          q,          r,  vtas, alpha,       beta,        phi, theta,     psi, he, xe, ye
                reset_up_bound = [deg2rad(0), deg2rad(0), 0,  0,    deg2rad(0),  deg2rad(0),  0, deg2rad(0),  0,   0, 0,  0];
                reset_lo_bound = [deg2rad(0), deg2rad(0), 0,  0,    deg2rad(0), deg2rad(0), 0, deg2rad(0), 0,  0, 0,  0];
            end
            % create a vector with randomized values within the bounds
            x_reset = zeros(1, length(obj.x0));
            for i = 1:length(obj.x0)
                x_reset(i) = reset_up_bound(i) + (reset_lo_bound(i) - ...
                    reset_up_bound(i)) * rand;
            end

            % convert x_reset to a string
            x_reset_str = mat2str(x_reset);
            reset_block_path = strcat(obj.mdl,"/Cessna Citation 500 " + ...
                "Non-Lineair 6 DOF model/Citation model DASMAT, " + ...
                "modified by Alexander Veldhuijzen and Clark " + ...
                "Borst/AIRCRAFT MODEL/EQM/x reset block");
            
            % update the value in the 'x reset block'
            set_param(reset_block_path, "Value", x_reset_str);
            obj.sm = setBlockParameter(obj.sm, obj.elevator_input, "value" ,num2str(0));
            % it is then summed with the 'x0 block' and fed as the initial
            % condition of the 'x integrator' block

            % output the initial state
            state_0 = x_reset + obj.x0;
            if obj.env_settings('longitudinal') == true
                %state_0 = state_0([2, 4, 5, 8, 10]);
                state_0 = state_0(obj.state_to_use);

            end
            if obj.sm.Status == "inactive"
                initialize(obj.sm);
                step(obj.sm, PauseTime= obj.sm.Time + obj.dt);
            end
            obj.counter = 1;
            % check current status
            obj.sm.Status
        end

        function stop(obj)
            % stop the simulation
            if ismember(obj.sm.Status, ["initialized", "running", "paused"])
                stop(obj.sm)
            end
        end

        function reward = reward_function(obj, ref_state, new_state)
            alpha = new_state(obj.state_to_use(1));
            q = new_state(obj.state_to_use(2));
            q_ref = ref_state(obj.counter+1);
            reward = -rad2deg((q-q_ref)^2);
        end

        function is_done = is_done_function(obj, new_state)
            % bounds for episode termination
             % bounds for episode termination
                %                   p,     q,    r,   vtas,  alpha, beta, phi, theta,         psi,  he,   xe,   ye
            % if obj.env_settings('longitudinal') == true
            %     state_up_limits = [ inf,  inf,  inf,  inf,  inf,   inf,  inf, deg2rad(40),   inf,  inf,  inf,  inf];
            %     state_lo_limits = [-inf, -inf, -inf, -inf, -inf,  -inf, -inf, deg2rad(-40), -inf, -inf, -inf, -inf];
            % else
            %     %                   p,     q,    r,   vtas, alpha,       beta,         phi,          theta,        psi,          he,    xe,   ye
            %     state_up_limits = [ inf,  inf,  inf, 300, deg2rad(40),  deg2rad(40),  deg2rad(80),  deg2rad(40),  deg2rad(80),  10000, inf,  inf];
            %     state_lo_limits = [-inf, -inf, -inf, 50,  deg2rad(-40), deg2rad(-40), deg2rad(-80), deg2rad(-40), deg2rad(-80), 0,    -inf, -inf];
            % end
            %check if limits have been exceeded
            up_limits_exceeded = new_state(obj.state_to_use) > obj.bounds.states_upper_bounds;
            lo_limits_exeeded = new_state(obj.state_to_use) < obj.bounds.states_lower_bounds;
            
            if any(up_limits_exceeded) || any(lo_limits_exeeded)
                is_done = true;
                disp('Terminating because a state limit is exceeded')
            else
                is_done = false;
            end

        end
    end
end