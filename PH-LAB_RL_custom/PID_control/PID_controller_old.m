classdef PID_controller < handle

    properties
        kp_i
        ki_i
        kd_i

        kp_o
        ki_o
        kd_o
        
        integral_i
        integral_o

        last_error_i
        last_error_o

        dt
        
        loop_i = []
        loop_o = []
    end


    methods
        function obj = PID_controller(dt, C_outer, C_inner)
            obj.dt = dt;
            if ~isempty(C_outer) && ~isempty(C_inner)
                %load ("C_inner.mat")
                obj.kp_i = -C_inner.Kp;
                obj.ki_i = -C_inner.Ki;
                obj.kd_i = -C_inner.Kd;
                %load("C_outer.mat")
                obj.kp_o = -C_outer.Kp;
                obj.ki_o = -C_outer.Ki;
                obj.kd_o = -C_outer.Kd;
            end
        end

        function action = control(obj, o_setpoint, o_val, i_val)
            % obj.kp_i = -10;
            % obj.ki_i = 0;
            % obj.kd_i = 0;
            % obj.kp_o = 1.5;
            % obj.ki_o = 0;
            % obj.kd_o = 0;
            control_signal_o = obj.outer_loop(o_setpoint, o_val);
            control_signal_i = obj.inner_loop(control_signal_o, i_val);
            action = -control_signal_i;
        end

        function control_signal = inner_loop(obj, setpoint, measured_val)
            error = setpoint - measured_val;
            obj.integral_i = obj.integral_i + error * obj.dt;
            derivative = (error - obj.last_error_i) / obj.dt;
            control_signal = obj.kp_i * error  + obj.ki_i * obj.integral_i + obj.kd_i * derivative;

            obj.last_error_i = error;
        end

        function control_signal = outer_loop(obj, setpoint, measured_val)
            error = setpoint - measured_val;
            obj.integral_o = obj.integral_o + error * obj.dt;
            derivative = (error - obj.last_error_o) / obj.dt;
            control_signal = obj.kp_o * error  + obj.ki_o * obj.integral_o + obj.kd_o * derivative;

            obj.last_error_o = error;
        end

        function reset(obj)
            obj.integral_i = 0;
            obj.integral_o = 0;
            obj.last_error_i = 0;
            obj.last_error_o = 0;
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