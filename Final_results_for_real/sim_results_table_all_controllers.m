%table for all the data
clear
clc
testCases = arrayfun(@(x) sprintf("$ bs text{TC-%d} $", x), 1:16, 'UniformOutput', false)';
groups = [repmat("Linear", 16, 1); repmat("CitAST", 16, 1)];
tc_full = [testCases; testCases];

% Placeholder values (you can replace these with actual data)
numRows = length(tc_full);
emptyCol = string(NaN(numRows, 1));

% Define table with appropriate variable names
T = table(...
    groups, ...
    tc_full, ...
    emptyCol, emptyCol, emptyCol, emptyCol, ...    % SAC-LSTM (Sine & Step)
    emptyCol, emptyCol, emptyCol, emptyCol, ...    % SAC-FF (Sine & Step)
    emptyCol, emptyCol, emptyCol, emptyCol ...     % LC (Sine & Step)
);
% Set column names
T.Properties.VariableNames = { ...
    'Group', 'TestCase', ...
    'SAC_LSTM_Sine_nMAE', 'SAC_LSTM_Sine_CA', ...
    'SAC_LSTM_Step_nMAE', 'SAC_LSTM_Step_CA', ...
    'SAC_FF_Sine_nMAE', 'SAC_FF_Sine_CA', ...
    'SAC_FF_Step_nMAE', 'SAC_FF_Step_CA', ...
    'LC_Sine_nMAE', 'LC_Sine_CA', ...
    'LC_Step_nMAE', 'LC_Step_CA' ...
};



load("FF_agent10_policy_37_all_TC_sim_data.mat")
load("LSTM_agent6_policy32_all_TC_sim_data.mat")
load("PID_sim_data.mat")
ref_type = ["Sine", "Step"];
env_type = ["Linear", "CitAST"];
agent_type = ["SAC_LSTM", "SAC_FF", "LC"];
% assigning the data to the table


for agent = 1:3
    if agent == 1
        agent_data = LSTM_agent6_policy32_all_TC_sim_data;
    elseif agent == 2
        agent_data = FF_agent10_policy_37_all_TC_sim_data;
    else
        agent_data = PID_sim_data;
    end
    for tc = 1:16
        for ref = 1:2
            tc_ref_data = agent_data{ref,tc};
            for env = 1:2
                if env == 1
                    env_data = tc_ref_data.lin_env;
                else
                    env_data = tc_ref_data.sl_env;
                end
                tc_env = 16*(env-1) + tc;

                nMAE = env_data.nMAE;
                CA = env_data.ctrl_activity;
                nMAE_var_name = sprintf("%s_%s_nMAE", agent_type(agent), ref_type(ref));
                CA_var_name = sprintf("%s_%s_CA", agent_type(agent), ref_type(ref));

                if 100*nMAE > 10
                    T{tc_env,nMAE_var_name} = sprintf("bs cellcolor[HTML]{FE996B} $ %0.1f bs_per $", 100*nMAE);
                else
                    T{tc_env,nMAE_var_name} = sprintf("$%0.1f bs_per $", 100*nMAE);
                end


                if CA > 0.1
                    T{tc_env,CA_var_name} = sprintf("bs cellcolor[HTML]{FE996B} $ bs SI{%0.1e}{} $", CA);
                else
                    T{tc_env,CA_var_name} = sprintf("$ bs SI{%0.1e}{} $", CA);
                end
            end
        end
    end
end

%writetable(T, 'controller_performance_table_v2.csv');
