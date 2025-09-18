% various configuration parameters of the RL environment


de_rate_lim = deg2rad(20); % limit is 20 deg/s
%de_rate_lim = deg2rad(100); % limit is 20 deg/s

da_rate_lim = deg2rad(20); % limit is 20 deg/s
dr_rate_lim = deg2rad(40); % limit is 40 deg/s

%de_up_sat_lim = deg2rad(15); % upper limit
%de_lo_sat_lim = deg2rad(-17); % lower limit
de_up_sat_lim = deg2rad(5); % upper limit
de_lo_sat_lim = deg2rad(-5); % lower limit

da_sat_lim = 0.5*deg2rad(22); % symmetric
dr_sat_lim = 0.5*deg2rad(34); % symmetric

p = 1;
q = 2;
r = 3;
alpha = 4;
V_TAS = 5;
Beta = 6;
phi = 7;
theta = 8;
psi = 9;
h = 10;

trimdatafile = 'full_DoF_v90ms_m4500kg_h2000m_q_init.tri';