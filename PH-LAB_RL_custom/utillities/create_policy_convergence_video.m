clearvars -except agents stats

Env_config
env_mdl = "Citation_RL_custom_env_by_Matan"; 
steps_per_episode = 1000;
dt = 0.01;
episode_length = steps_per_episode * dt;
time = (0:dt:episode_length)';
q_ref = deg2rad(5)*sin(2*pi*0.2*time)'; % sine wave
env_settings = containers.Map({'ref_state',  'dt', 'longitudinal', 'linear', 'steps_per_episode', 'RF'}, ...
                              {q_ref,         dt,   true,           true,     steps_per_episode,   1});
env = Linear_Citation_env(env_settings);
sac = SAC(env);
mesh_densidy = 51;
%alpha = linspace(-deg2rad(20), deg2rad(20), 10000); 
range = linspace(-deg2rad(20), deg2rad(20), mesh_densidy);
states = [range', range', range'];

[alpha, q] = meshgrid(states(:,1));


for pol=1:100
    pol
    action = zeros(size(alpha));
    for i = 1:length(alpha)
        action(i,:) = sac.predict([alpha(i,:)', q(i,:)', zeros(length(range),1)], true, agents(1,2).datalogger.actor.policy.net(pol));
        % action(i,:) = sac.predict([alpha(i,:)', q(i,:)', q(i,:)'], true, datalogger.actor.policy.net(best_RF_idx));

    end
    RF_actions{pol} = action;
end


% figure
% s = surf(alpha, q, action);
% xlabel('alpha [rad] (normalized)')
% ylabel('q [rad/s] (normalized)')
% zlabel('elevator deflection [rad]')

%%

PS = PLOT_STANDARDS();
angle1 = linspace(0,45,100)
for i = 1:3
    color_map(:,i) = linspace(PS.MyBlue2(i), PS.MyRed(i), mesh_densidy)
end

video_writer = VideoWriter('avi');
video_writer.FrameRate = 8;
open(video_writer);
    
    % Create figure for rendering
fig = figure('Position', [100, 100, 1200, 800]);
frames = [1*ones(1,12),...
          2*ones(1,10),...
          3*ones(1,8),...
          4*ones(1,6),...
          5*ones(1,4),...
          6*ones(1,2)]
frames = [frames, frames(end):100]
angle1 = linspace(90,45,length(frames))
angle2 = linspace(0,45,length(frames))


    % Iterate through episodes and create surface plots
for i = 1:length(frames)
    pol = frames(i)

    % Clear previous plot
    clf;
    
    % Get policy for current episode
    % Create surface plot
    s = surf(alpha, q, RF_actions{pol});
    
    % Customize plot
    %colormap(color_map);
    %colorbar;
    %shading interp;
    %view([45, 45]);

    view([angle1(i), angle2(i)])

    % Add title and labels
    title(sprintf('RF2 Policy %d/100', pol), 'FontSize', 20);
    xlabel('\alpha [rad]', 'FontSize', 18);
    ylabel('q [rad/s]', 'FontSize', 18);
    zlabel('\delta_e', 'FontSize', 18);
    
    % Adjust axis limits for consistency across episodes
    zlim('manual');
    if pol == 1
        initial_zlim = zlim;
    end
    zlim([-0.5 0.5]);
    
    % Capture frame
    frame = getframe(fig);
    writeVideo(video_writer, frame);
end

% Close video writer
close(video_writer);
close(fig);
