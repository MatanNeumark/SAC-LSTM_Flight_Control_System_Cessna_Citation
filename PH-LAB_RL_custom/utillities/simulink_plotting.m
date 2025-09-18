
%% PID data for experiment I

PID_for_exp2_step.alpha = longitudinal_state.signals.values(:,1);
PID_for_exp2_step.q = longitudinal_state.signals.values(:,2);
PID_for_exp2_step.q_ref = ref_signal;
PID_for_exp2_step.elevator = elevator;

q_error = PID_for_exp2_step.q_ref.Data - PID_for_exp2_step.q;
delta_action = PID_for_exp2_step.elevator.Data(2:end) - PID_for_exp2_step.elevator.Data(1:end-1);
PID_for_exp2_step.nMAE = mean(abs(q_error)) / mean(abs(PID_for_exp2_step.q_ref.Data));
PID_for_exp2_step.MTV = mean(abs(delta_action));
figure
hold on
plot((longitudinal_state.signals.values(:,1)), LineWidth=2)
plot((longitudinal_state.signals.values(:,2)), 'magenta', LineWidth=2)
plot((ref_signal.Data), 'k', LineStyle="--", LineWidth=2)
plot((elevator.data), Color=[0.7, 0.7, 0], LineWidth=1)
yline(0)
grid on
%ylim([- 1.2, 1.2])
xlim([0,1000])
xlabel('time steps, dt=0.01')
ylabel('[deg]')
legend('\alpha', 'q', 'q_{ref}', '\delta_e')
title("PID, nMAE = " + 100*PID_for_exp2_step.nMAE + "%")


%% PID data for experiment II


%% PID data for experiment III

