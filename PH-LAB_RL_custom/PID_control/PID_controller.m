classdef PID_controller < handle

    properties
        kp_i
        ki_i
        kd_i
        N_i

        kp_o
        ki_o
        kd_o
        N_o
        
        dt
        
        loop_i = []
        loop_o = []
        
        mdl
        theta_ref
        theta
        q
        sm
        ep_length
    end


    methods
        function obj = PID_controller(dt, steps_per_episode, C_outer, C_inner)
            obj.dt = dt;
            if ~isempty(C_outer) && ~isempty(C_inner)
                %load ("C_inner.mat")
                obj.kp_i = C_inner.Kp;
                obj.ki_i = C_inner.Ki;
                obj.kd_i = C_inner.Kd;
                obj.N_i = C_inner.N;
                %load("C_outer.mat")
                obj.kp_o = C_outer.Kp;
                obj.ki_o = C_outer.Ki;
                obj.kd_o = C_outer.Kd;
                obj.N_o = C_outer.N;
            end
            obj.ep_length = 0: obj.dt :obj.dt*steps_per_episode;
            %obj.ep_length = obj.dt*steps_per_episode;
            obj.mdl = 'PID_blocks';
            assignin('base','Ts',obj.dt);
            assignin('base','Tf',obj.ep_length);
            assignin("base", 'kp_o', obj.kp_o)
            assignin("base", 'ki_o', obj.ki_o)
            assignin("base", 'kd_o', obj.kd_o)
            assignin("base", 'N_o', obj.N_o)
            assignin("base", 'kp_i', obj.kp_i)
            assignin("base", 'ki_i', obj.ki_i)
            assignin("base", 'kd_i', obj.kd_i)
            assignin("base", 'N_i', obj.N_i)

            obj.theta_ref = strcat(obj.mdl,"/theta_ref");
            obj.theta = strcat(obj.mdl,"/theta");
            obj.q = strcat(obj.mdl,"/q");
            %open_system(mdl);
            obj.sm = simulation(obj.mdl);
            obj.sm = setModelParameter(obj.sm, StopTime=num2str(obj.ep_length(end) + obj.dt));
            %obj.sm = setModelParameter(obj.sm, StopTime=num2str(obj.ep_length(end)));
            obj.sm.Status

        end

        function action = control(obj, q_val, theta_val, theta_ref_val)
            if obj.sm.Status == "inactive"
               state_0 = obj.reset;
            end
            obj.sm = setBlockParameter(obj.sm, obj.theta_ref, "value" ,num2str(theta_ref_val));
            obj.sm = setBlockParameter(obj.sm, obj.theta, "value" ,num2str(theta_val));
            obj.sm = setBlockParameter(obj.sm, obj.q, "value" ,num2str(q_val));
            step(obj.sm, PauseTime= obj.sm.Time + obj.dt);
            action = obj.sm.SimulationOutput.yout(end,:);

        end

        function reset(obj)
            if ismember(obj.sm.Status, ["initialized", "running", "paused"])
                stop(obj.sm);
            end
            obj.sm = setBlockParameter(obj.sm, obj.theta_ref, "value" ,num2str(0));
            obj.sm = setBlockParameter(obj.sm, obj.theta, "value" ,num2str(0));
            obj.sm = setBlockParameter(obj.sm, obj.q, "value" ,num2str(0));
            if obj.sm.Status == "inactive"
                initialize(obj.sm);
                step(obj.sm, PauseTime= obj.sm.Time + obj.dt);
            end
            obj.sm.Status
        end
        
        function stop(obj)
            % stop the simulation
            if ismember(obj.sm.Status, ["initialized", "running", "paused"])
                stop(obj.sm)
            end
        end

        function [C_outer, C_inner] = auto_tune(obj, env_ss, servo_ss, pm_outer, pm_inner, cf_outer, cf_inner, verbose)
            if ~isempty(servo_ss)
                env_tf = tf(env_ss) * tf(servo_ss);
            else
                env_tf = tf(env_ss);
            end
            inner_tf = env_tf(1); % q
            outer_tf = env_tf(3); % theta

            opts_i = pidtuneOptions('PhaseMargin', pm_inner, 'DesignFocus', 'reference-tracking');
            opts_o = pidtuneOptions('PhaseMargin', pm_outer, 'DesignFocus', 'reference-tracking');

            [C_inner, info_i] = pidtune(inner_tf,'PID',cf_inner, opts_i);
            info_i
            obj.loop_i  = feedback(C_inner*inner_tf, 1, -1);
            if verbose
                figure
                step(obj.loop_i)
            end
             
            [C_outer, info_o] = pidtune(outer_tf*obj.loop_i, 'PID', cf_outer, opts_o);
            info_o
            obj.loop_o  = feedback(C_outer*outer_tf*obj.loop_i, 1, -1);
            if verbose
                figure
                step(obj.loop_o)
            end
        end

        function [C_outer, C_inner] = manual_tune(obj, env_ss, servo_ss, kp_o, ki_o, kd_o, kp_i, ki_i, kd_i, Tf, verbose)
            if ~isempty(servo_ss)
                env_tf = tf(env_ss) * tf(servo_ss);
            else
                env_tf = tf(env_ss);
            end
            inner_tf = env_tf(1); % q
            outer_tf = env_tf(3); % theta
            figure
            margin(inner_tf)
            C_inner = pid(-kp_i, -ki_i, -kd_i, Tf);
            
            obj.loop_i  = feedback(C_inner*inner_tf, 1, -1);
            if verbose
                figure
                step(obj.loop_i)
                figure
                bode(obj.loop_i)
                figure
                margin(obj.loop_i)

            end
             
            C_outer = pid(-kp_o, -ki_o, -kd_o)
           
            obj.loop_o  = feedback(C_outer*outer_tf*obj.loop_i, 1, -1);
            if verbose
                figure
                step(obj.loop_o)
                figure
                bode(obj.loop_o)
                figure
                margin(obj.loop_o)
            end
        end

        function sim(obj, ctrl_ref, time)
            y = lsim(obj.loop_o, ctrl_ref, time);
            figure
            hold on
            grid on
            plot(time, y, LineWidth=1)
            plot(time, ctrl_ref, LineWidth=1, LineStyle="--")
            legend('response', 'state_{ref}')
        end
    end



end