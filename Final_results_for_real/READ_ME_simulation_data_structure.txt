Data structure for:
	LSTM_agent6_policy32_all_TC_sim_data
	FF_agent10_policy37_all_TC_sim_data

struct is 2x16:
	16 combinations of: "noise and bias", "delay", "servo rate limit", "servo transfer function".
	2 reference signals: sine 5*(pi/180)*sin(0.2*pi*t) and step with amplitude of 5 degrees after 200 time steps.

Then, each cell is 2x1:
	linear-dynamics environment
	nonlinear dynamics environment (CitAST in Simulink)
