clc
sac = SAC(env);
%alpha = linspace(-deg2rad(20), deg2rad(20), 10000); 
range = linspace(-deg2rad(20), deg2rad(20), 101);
states = [range', range', range'];

[alpha, q] = meshgrid(states(:,1));
action = zeros(size(alpha));

RF=5;
idx = exp1.best_idx;
best_RF2_idx = stats.best_policy_idx(idx(RF),RF);
datalogger = agents(idx(RF),RF).datalogger;
for i = 1:length(alpha)
        %action(i,:) = sac.predict([alpha(i,:)', q(i,:)', q(i,:)'], true, datalogger.actor.policy.net(best_RF2_idx))
        %ction(i,:) = sac.predict([alpha(i,:)', q(i,:)', q(i,:)'], true, RLTb_reward6_seed_91.actorNet)
        %action(i,:) = sac.predict([alpha(i,:)', q(i,:)', zeros(length(range),1)], true, datalogger.actor.policy.net(best_RF2_idx));
        action(i,:) = sac.predict([alpha(i,:)', q(i,:)', zeros(length(range),1)], true, RLTb_reward6_seed_91.actorNet);

        %action(i,:) = sac.predict([alpha(i,:)', q(i,:)', q(i,:)'], true, actorNet);
end

figure
s = surf(alpha, q, action);
xlabel('alpha [rad] (normalized)')
ylabel('q [rad/s] (normalized)')
zlabel('elevator deflection [rad]')

figure
hold on
for i = 1:101
    plot(alpha(1,:), action(i,:))
    ylim([-0.5 0.5])
    %scatter(alpha(1,:), action(e(i),:), 5, 'filled')
    %scatter(alpha, log_sigma, 10, 'filled')
    % scatter(alpha, sigma, 10, 'filled')
    %plot(alpha, scaled_action)
end
xline(0)
yline(0)
ylabel('elevator deflection [rad]')
xlabel('alpha [rad] (normalized)')
title('Policy response as a function of alpha (x axis) and q (lines)')
grid on
hold off

figure
hold on
for i = 1:101
    plot(q(:,1), action(:,i))
    ylim([-0.5 0.5])
    %scatter(alpha(1,:), action(e(i),:), 5, 'filled')
    %scatter(alpha, log_sigma, 10, 'filled')
    % scatter(alpha, sigma, 10, 'filled')
    %plot(alpha, scaled_action)
end
xline(0)
yline(0)
ylabel('elevator deflection [rad]')
xlabel('q [rad] (normalized)')
title('Policy response as a function of q (x axis) and alpha (lines)')
grid on
hold off

