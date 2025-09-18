clear
steps_per_episode = 1000;
total_steps = 1e5;%1.6e5;
dt = 0.01;
episode_length = steps_per_episode * dt;
n_par_envs = 6;
buffer_size = 1e4;%5e4;
minibatch_size = 64;%250; %256
num_epoch = 2;%4;
start_steps = 0;
update_freq = steps_per_episode;
t = zeros(n_par_envs,100);
for n = 1:n_par_envs
    buffer_size = 1e4;
    num_epoch = 2*n;
    iter = 0;
    i = 1;
    while t(n,i) < total_steps
        i = i + 1;
        t(n,i) = t(n,i-1) + steps_per_episode*n;
        % if t(n,i)>total_steps
        %     t(n,i)=total_steps;
        % end
        if t(n,i) > buffer_size
            num_minibatch_per_epoch = floor(buffer_size / minibatch_size);
        else
            num_minibatch_per_epoch = floor(t(n,i) / minibatch_size);
        end
        for epoch = 1:num_epoch
            for iteration = 1:num_minibatch_per_epoch
                iter = iter + 1;
            end
        end
        n_iter(n,i) = iter;
    end
    t_plot{n} = t(n,1:i);
    n_iter_plot{n} = n_iter(n,1:i);
end

figure
hold on
grid on
for n = 1:n_par_envs
    plot(t_plot{n}, n_iter_plot{n}, LineWidth=1)
end
legend('1 env', '2 env', '3 env', '4 env', '5 env', '6 env')

%%
% lr_init = 1e-2;
% lr_final = 1e-4;
% total_steps = 4e5;
lr_init = 1e-2;
lr_final = 1e-5;
total_steps = 4e5;
lr_discount = (lr_final/lr_init)^(1/total_steps);
t = 1:total_steps;
lr = lr_discount.^t * lr_init;

%figure
hold on
grid on
plot(lr)