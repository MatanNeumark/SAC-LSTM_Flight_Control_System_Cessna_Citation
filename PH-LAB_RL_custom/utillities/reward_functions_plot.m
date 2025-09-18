
error = linspace(0.1, 0, 10000);
action = linspace(-0.5, 0, 10000);
previous_action = linspace(0.5, 0, 10000);

RF1 = -(error).^2;
RF2 = -(180/pi)*(error).^2;
RF3 = -abs(error);
RF4 = -(abs(error) + 0.1*abs(action - previous_action));
%RF5 = -tanh(abs(rad2deg(error)));

clear fig4_comps
figure(4)
fig4_comps.fig = gcf;
hold on
grid on
fig4_comps.p1 = plot(error, RF1, Color=PS.MyBlue4, LineWidth=1.5);
fig4_comps.p2 = plot(error, RF2, Color=PS.MyGreen4, LineWidth=1.5);
fig4_comps.p3 = plot(error, RF3, Color=PS.Orange3, LineWidth=1.5);
fig4_comps.p4 = plot(error, RF4, Color=PS.MyPink, LineWidth=1.5);
%title("Mean of normalized returns")
legend('RF1= -e^2', 'RF2= -(180/\pi)e^2', 'RF3= -|e|', 'RF4= -|e| -0.1|\Delta\delta_e|', "Position", [0.14 0.27 0.24 0.15])
xlabel('Pitch rate error [rad/s]')
ylabel('Reward')
ylim([-0.2 0])
% xlim([0,0.1])
pbaspect([2 1 1])
STANDARDIZE_FIGURE(fig4_comps)
SAVE_MY_FIGURE(fig4_comps, 'exp1_reward_functions.png', 'small');