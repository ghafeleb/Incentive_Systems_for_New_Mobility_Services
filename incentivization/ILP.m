function ILP(iterRun, nonuser_perc_prob, budget, n_iter_ADMM, ...
            n_time, n_time_inc_start_ADMM, n_time_inc_end_ADMM, ...
            VOT, seedADMM, seedData, rho,...
            fairness, step_size, region_, setting_region, ...
            MIPGap)
% # of companies of ADMM based algorithm. It should stay 1. 
% Assignment of drivers to companies happens randomly after solving the problem.
n_companies_ADMM = 1; 
iterRun = iterRun+1;
nonuser_prob = nonuser_perc_prob/100;
% Percentace of nonuser drivers in each time interval
nonuser_perc_ADMM = repmat(nonuser_prob, n_time, 1);
setting_output_ADMM = sprintf('%.0f', nonuser_perc_ADMM(1)*100);
LogicalStr = {'F', 'T'};
initializeSBaseline = true;
fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})

startTime0 = tic; % Start time of running time of whole code

%% Load data
inputFolder0 = fullfile('../data', region_, setting_region);
folderRun = fullfile(inputFolder0, ...
                    strcat('Det_initAll2_MultT', ...
                    '_b', num2str(budget), ...
                    '_sD', num2str(seedData), ...
                    '_sA', num2str(seedADMM), ...
                    '_r', num2str(rho), ...
                    '_it', num2str(n_iter_ADMM),...
                    '_VOT', num2str(VOT), ...
                    '_nC', num2str(n_companies_ADMM), ...
                    '_f', fairness, ...
                    '_initSB_', LogicalStr{initializeSBaseline + 1}, ...
                    '_percNonU', setting_output_ADMM, ...
                    '_nTIS', num2str(n_time_inc_start_ADMM), ...
                    '_nTIE', num2str(n_time_inc_end_ADMM), ...
                    '_ss', num2str(step_size), ...
                    '_itN', num2str(iterRun)));
fileADMM = fullfile(folderRun, strcat('result_', num2str(iterRun), '.mat'));
% load(folderRun, 'n_driver_company_time', 'idxEndCell');
load(fileADMM, 'tt0_array*', 'w_array*', 'n_OD', 'R_matrix', 'D', ...
    'ones_S', 'v_tilda_baseline', 'RHS', 'idxEndCell', 'gamma', ...
    'n_path', 'n_driver_company_time', 'link_loc', 'q_company_time', ...
    'delta_p_t', 'alpha', 'num_path_v', 'n_link', ...
    'v_dynamic_array_NoInc_allDet', 'S', 'n_driver_company', 'q_company', ...
    'B', 'etaDet', 'delta_pDet', 'c_tilda', 'q', 'q_nonuser', 'q_user')

%% Log outputs on Command Window
log_file = fullfile(folderRun, strcat('outputLog_MIPGap', num2str(MIPGap), '_ILP.txt'));
diary(log_file); % Start logging to a file named 'outputLog.txt'

%% ILP
time_LP_start = tic;
cvx_solver Gurobi_2
cvx_solver_settings('MIPGap', MIPGap)
cvx_begin
variable S11_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+0}) binary;
variable S12_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+1}) binary;
variable S13_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+2}) binary;
variable S14_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+3}) binary;
variable S15_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+4}) binary;
variable S16_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+5}) binary;
variable S17_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+6}) binary;
variable S18_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+7}) binary;
variable S19_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+8}) binary;
variable S110_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+9}) binary;
variable S111_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+10}) binary;
variable S112_binary(n_path, n_driver_company_time{1, n_time_inc_start_ADMM+11}) binary;
variable c(1, 1);

% minimize(ones(1, n_path)*(abs(S11_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM}, 1) - ...
%     S{1}((n_time_inc_start_ADMM-1+0)*n_path+1:n_time_inc_start_ADMM*n_path, 1:idxEndCell{1, n_time_inc_start_ADMM+0})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+0}, 1))) + ...
%     ones(1, n_path)*(abs(S12_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+1}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+1)*n_path+1:(n_time_inc_start_ADMM+1)*n_path, idxEndCell{1, n_time_inc_start_ADMM+0}+1:idxEndCell{1, n_time_inc_start_ADMM+1})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+1}, 1))) + ...
%     ones(1, n_path)*(abs(S13_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+2}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+2)*n_path+1:(n_time_inc_start_ADMM+2)*n_path, idxEndCell{1, n_time_inc_start_ADMM+1}+1:idxEndCell{1, n_time_inc_start_ADMM+2})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+2}, 1))) + ...
%     ones(1, n_path)*(abs(S14_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+3}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+3)*n_path+1:(n_time_inc_start_ADMM+3)*n_path, idxEndCell{1, n_time_inc_start_ADMM+2}+1:idxEndCell{1, n_time_inc_start_ADMM+3})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+3}, 1))) + ...
%     ones(1, n_path)*(abs(S15_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+4}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+4)*n_path+1:(n_time_inc_start_ADMM+4)*n_path, idxEndCell{1, n_time_inc_start_ADMM+3}+1:idxEndCell{1, n_time_inc_start_ADMM+4})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+4}, 1))) + ...
%     ones(1, n_path)*(abs(S16_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+5}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+5)*n_path+1:(n_time_inc_start_ADMM+5)*n_path, idxEndCell{1, n_time_inc_start_ADMM+4}+1:idxEndCell{1, n_time_inc_start_ADMM+5})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+5}, 1))) + ...
%     ones(1, n_path)*(abs(S17_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+6}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+6)*n_path+1:(n_time_inc_start_ADMM+6)*n_path, idxEndCell{1, n_time_inc_start_ADMM+5}+1:idxEndCell{1, n_time_inc_start_ADMM+6})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+6}, 1))) + ...
%     ones(1, n_path)*(abs(S18_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+7}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+7)*n_path+1:(n_time_inc_start_ADMM+7)*n_path, idxEndCell{1, n_time_inc_start_ADMM+6}+1:idxEndCell{1, n_time_inc_start_ADMM+7})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+7}, 1))) + ...
%     ones(1, n_path)*(abs(S19_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+8}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+8)*n_path+1:(n_time_inc_start_ADMM+8)*n_path, idxEndCell{1, n_time_inc_start_ADMM+7}+1:idxEndCell{1, n_time_inc_start_ADMM+8})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+8}, 1))) + ...
%     ones(1, n_path)*(abs(S110_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+9}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+9)*n_path+1:(n_time_inc_start_ADMM+9)*n_path, idxEndCell{1, n_time_inc_start_ADMM+8}+1:idxEndCell{1, n_time_inc_start_ADMM+9})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+9}, 1))) + ...
%     ones(1, n_path)*(abs(S111_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+10}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+10)*n_path+1:(n_time_inc_start_ADMM+10)*n_path, idxEndCell{1, n_time_inc_start_ADMM+9}+1:idxEndCell{1, n_time_inc_start_ADMM+10})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+10}, 1))) + ...
%     ones(1, n_path)*(abs(S112_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+11}, 1) - ...
%     S{1}(((n_time_inc_start_ADMM-1)+11)*n_path+1:(n_time_inc_start_ADMM+11)*n_path, idxEndCell{1, n_time_inc_start_ADMM+10}+1:idxEndCell{1, n_time_inc_start_ADMM+11})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+11}, 1))) + ...
%     0);
minimize(sum(abs(S11_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM}, 1) - ...
    S{1}((n_time_inc_start_ADMM-1+0)*n_path+1:n_time_inc_start_ADMM*n_path, 1:idxEndCell{1, n_time_inc_start_ADMM+0})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+0}, 1))) + ...
    sum(abs(S12_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+1}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+1)*n_path+1:(n_time_inc_start_ADMM+1)*n_path, idxEndCell{1, n_time_inc_start_ADMM+0}+1:idxEndCell{1, n_time_inc_start_ADMM+1})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+1}, 1))) + ...
    sum(abs(S13_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+2}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+2)*n_path+1:(n_time_inc_start_ADMM+2)*n_path, idxEndCell{1, n_time_inc_start_ADMM+1}+1:idxEndCell{1, n_time_inc_start_ADMM+2})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+2}, 1))) + ...
    sum(abs(S14_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+3}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+3)*n_path+1:(n_time_inc_start_ADMM+3)*n_path, idxEndCell{1, n_time_inc_start_ADMM+2}+1:idxEndCell{1, n_time_inc_start_ADMM+3})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+3}, 1))) + ...
    sum(abs(S15_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+4}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+4)*n_path+1:(n_time_inc_start_ADMM+4)*n_path, idxEndCell{1, n_time_inc_start_ADMM+3}+1:idxEndCell{1, n_time_inc_start_ADMM+4})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+4}, 1))) + ...
    sum(abs(S16_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+5}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+5)*n_path+1:(n_time_inc_start_ADMM+5)*n_path, idxEndCell{1, n_time_inc_start_ADMM+4}+1:idxEndCell{1, n_time_inc_start_ADMM+5})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+5}, 1))) + ...
    sum(abs(S17_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+6}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+6)*n_path+1:(n_time_inc_start_ADMM+6)*n_path, idxEndCell{1, n_time_inc_start_ADMM+5}+1:idxEndCell{1, n_time_inc_start_ADMM+6})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+6}, 1))) + ...
    sum(abs(S18_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+7}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+7)*n_path+1:(n_time_inc_start_ADMM+7)*n_path, idxEndCell{1, n_time_inc_start_ADMM+6}+1:idxEndCell{1, n_time_inc_start_ADMM+7})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+7}, 1))) + ...
    sum(abs(S19_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+8}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+8)*n_path+1:(n_time_inc_start_ADMM+8)*n_path, idxEndCell{1, n_time_inc_start_ADMM+7}+1:idxEndCell{1, n_time_inc_start_ADMM+8})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+8}, 1))) + ...
    sum(abs(S110_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+9}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+9)*n_path+1:(n_time_inc_start_ADMM+9)*n_path, idxEndCell{1, n_time_inc_start_ADMM+8}+1:idxEndCell{1, n_time_inc_start_ADMM+9})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+9}, 1))) + ...
    sum(abs(S111_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+10}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+10)*n_path+1:(n_time_inc_start_ADMM+10)*n_path, idxEndCell{1, n_time_inc_start_ADMM+9}+1:idxEndCell{1, n_time_inc_start_ADMM+10})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+10}, 1))) + ...
    sum(abs(S112_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+11}, 1) - ...
    S{1}(((n_time_inc_start_ADMM-1)+11)*n_path+1:(n_time_inc_start_ADMM+11)*n_path, idxEndCell{1, n_time_inc_start_ADMM+10}+1:idxEndCell{1, n_time_inc_start_ADMM+11})*ones(n_driver_company_time{1, n_time_inc_start_ADMM+11}, 1))) + ...
    0);
subject to
D((n_time_inc_start_ADMM-1+0)*n_OD+1:n_OD*(n_time_inc_start_ADMM+0), ((n_time_inc_start_ADMM-1)+0)*n_path+1:(n_time_inc_start_ADMM+0)*n_path)*S11_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+0}, 1) == q_company_time{1, n_time_inc_start_ADMM+0};
D((n_time_inc_start_ADMM-1+1)*n_OD+1:n_OD*(n_time_inc_start_ADMM+1), ((n_time_inc_start_ADMM-1)+1)*n_path+1:(n_time_inc_start_ADMM+1)*n_path)*S12_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+1}, 1) == q_company_time{1, n_time_inc_start_ADMM+1};
D((n_time_inc_start_ADMM-1+2)*n_OD+1:n_OD*(n_time_inc_start_ADMM+2), ((n_time_inc_start_ADMM-1)+2)*n_path+1:(n_time_inc_start_ADMM+2)*n_path)*S13_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+2}, 1) == q_company_time{1, n_time_inc_start_ADMM+2};
D((n_time_inc_start_ADMM-1+3)*n_OD+1:n_OD*(n_time_inc_start_ADMM+3), ((n_time_inc_start_ADMM-1)+3)*n_path+1:(n_time_inc_start_ADMM+3)*n_path)*S14_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+3}, 1) == q_company_time{1, n_time_inc_start_ADMM+3};
D((n_time_inc_start_ADMM-1+4)*n_OD+1:n_OD*(n_time_inc_start_ADMM+4), ((n_time_inc_start_ADMM-1)+4)*n_path+1:(n_time_inc_start_ADMM+4)*n_path)*S15_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+4}, 1) == q_company_time{1, n_time_inc_start_ADMM+4};
D((n_time_inc_start_ADMM-1+5)*n_OD+1:n_OD*(n_time_inc_start_ADMM+5), ((n_time_inc_start_ADMM-1)+5)*n_path+1:(n_time_inc_start_ADMM+5)*n_path)*S16_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+5}, 1) == q_company_time{1, n_time_inc_start_ADMM+5};
D((n_time_inc_start_ADMM-1+6)*n_OD+1:n_OD*(n_time_inc_start_ADMM+6), ((n_time_inc_start_ADMM-1)+6)*n_path+1:(n_time_inc_start_ADMM+6)*n_path)*S17_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+6}, 1) == q_company_time{1, n_time_inc_start_ADMM+6};
D((n_time_inc_start_ADMM-1+7)*n_OD+1:n_OD*(n_time_inc_start_ADMM+7), ((n_time_inc_start_ADMM-1)+7)*n_path+1:(n_time_inc_start_ADMM+7)*n_path)*S18_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+7}, 1) == q_company_time{1, n_time_inc_start_ADMM+7};
D((n_time_inc_start_ADMM-1+8)*n_OD+1:n_OD*(n_time_inc_start_ADMM+8), ((n_time_inc_start_ADMM-1)+8)*n_path+1:(n_time_inc_start_ADMM+8)*n_path)*S19_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+8}, 1) == q_company_time{1, n_time_inc_start_ADMM+8};
D((n_time_inc_start_ADMM-1+9)*n_OD+1:n_OD*(n_time_inc_start_ADMM+9), ((n_time_inc_start_ADMM-1)+9)*n_path+1:(n_time_inc_start_ADMM+9)*n_path)*S110_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+9}, 1) == q_company_time{1, n_time_inc_start_ADMM+9};
D((n_time_inc_start_ADMM-1+10)*n_OD+1:n_OD*(n_time_inc_start_ADMM+10), ((n_time_inc_start_ADMM-1)+10)*n_path+1:(n_time_inc_start_ADMM+10)*n_path)*S111_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+10}, 1) == q_company_time{1, n_time_inc_start_ADMM+10};
D((n_time_inc_start_ADMM-1+11)*n_OD+1:n_OD*(n_time_inc_start_ADMM+11), ((n_time_inc_start_ADMM-1)+11)*n_path+1:(n_time_inc_start_ADMM+11)*n_path)*S112_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+11}, 1) == q_company_time{1, n_time_inc_start_ADMM+11};
S11_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+0}, 1);
S12_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+1}, 1);
S13_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+2}, 1);
S14_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+3}, 1);
S15_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+4}, 1);
S16_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+5}, 1);
S17_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+6}, 1);
S18_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+7}, 1);
S19_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+8}, 1);
S110_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+9}, 1);
S111_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+10}, 1);
S112_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_ADMM+11}, 1);
S11_binary'*delta_p_t{n_time_inc_start_ADMM+0} <= RHS{1}(1:idxEndCell{1, n_time_inc_start_ADMM});
S12_binary'*delta_p_t{n_time_inc_start_ADMM+1} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+1}+1:idxEndCell{1, n_time_inc_start_ADMM+1});
S13_binary'*delta_p_t{n_time_inc_start_ADMM+2} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+2}+1:idxEndCell{1, n_time_inc_start_ADMM+2});
S14_binary'*delta_p_t{n_time_inc_start_ADMM+3} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+3}+1:idxEndCell{1, n_time_inc_start_ADMM+3});
S15_binary'*delta_p_t{n_time_inc_start_ADMM+4} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+4}+1:idxEndCell{1, n_time_inc_start_ADMM+4});
S16_binary'*delta_p_t{n_time_inc_start_ADMM+5} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+5}+1:idxEndCell{1, n_time_inc_start_ADMM+5});
S17_binary'*delta_p_t{n_time_inc_start_ADMM+6} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+6}+1:idxEndCell{1, n_time_inc_start_ADMM+6});
S18_binary'*delta_p_t{n_time_inc_start_ADMM+7} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+7}+1:idxEndCell{1, n_time_inc_start_ADMM+7});
S19_binary'*delta_p_t{n_time_inc_start_ADMM+8} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+8}+1:idxEndCell{1, n_time_inc_start_ADMM+8});
S110_binary'*delta_p_t{n_time_inc_start_ADMM+9} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+9}+1:idxEndCell{1, n_time_inc_start_ADMM+9});
S111_binary'*delta_p_t{n_time_inc_start_ADMM+10} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+10}+1:idxEndCell{1, n_time_inc_start_ADMM+10});
S112_binary'*delta_p_t{n_time_inc_start_ADMM+11} <= RHS{1}(idxEndCell{1, (n_time_inc_start_ADMM-1)+11}+1:idxEndCell{1, n_time_inc_start_ADMM+11});
c(1) >= alpha(1).*(delta_p_t{n_time_inc_start_ADMM}'*S11_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM}, 1) +...
    delta_p_t{n_time_inc_start_ADMM+1}'*S12_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+1}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+2}'*S13_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+2}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+3}'*S14_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+3}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+4}'*S15_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+4}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+5}'*S16_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+5}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+6}'*S17_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+6}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+7}'*S18_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+7}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+8}'*S19_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+8}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+9}'*S110_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+9}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+10}'*S111_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+10}, 1) + ...
    delta_p_t{n_time_inc_start_ADMM+11}'*S112_binary*ones(n_driver_company_time{1, n_time_inc_start_ADMM+11}, 1) + ...
    - gamma(1));
c(1) >= 0;
c(1) + ...
    0 <= budget;
% idxEndRow1 = 0;
% idxStartCol = cell(n_companies_ADMM, n_time);
% idxStartCol(:) = {1};
% idxEndCol = cell(n_companies_ADMM, n_time);
% idxEndCol(:) = {0};
% nDriverTemp = cell(n_companies_ADMM, n_time);
% for iter_OD=1:n_OD
%     nPathTemp = num_path_v(iter_OD, 1);
%     for iter_company=1:n_companies_ADMM
%         for iter_time=1:n_time
%             nDriverTemp{iter_company, iter_time}  = q_company_time{iter_company, iter_time}(iter_OD, 1);
%         end
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+0}>0
%         idxEndCol{1, n_time_inc_start_ADMM+0} = idxEndCol{1, n_time_inc_start_ADMM+0} + nDriverTemp{1, n_time_inc_start_ADMM+0};
%         if iter_OD==1
%             S11_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+0}) == 0;
%         elseif iter_OD == n_OD
%             S11_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+0}:end) == 0;
%         else
%             S11_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+0}:idxEndCol{1, n_time_inc_start_ADMM+0}) == 0;
%             S11_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+0}:idxEndCol{1, n_time_inc_start_ADMM+0}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+0} = idxEndCol{1, n_time_inc_start_ADMM+0} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+1}>0
%         idxEndCol{1, n_time_inc_start_ADMM+1} = idxEndCol{1, n_time_inc_start_ADMM+1} + nDriverTemp{1, n_time_inc_start_ADMM+1};
%         if iter_OD==1
%             S12_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+1}) == 0;
%         elseif iter_OD == n_OD
%             S12_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+1}:end) == 0;
%         else
%             S12_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+1}:idxEndCol{1, n_time_inc_start_ADMM+1}) == 0;
%             S12_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+1}:idxEndCol{1, n_time_inc_start_ADMM+1}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+1} = idxEndCol{1, n_time_inc_start_ADMM+1} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+2}>0
%         idxEndCol{1, n_time_inc_start_ADMM+2} = idxEndCol{1, n_time_inc_start_ADMM+2} + nDriverTemp{1, n_time_inc_start_ADMM+2};
%         if iter_OD==1
%             S13_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+2}) == 0;
%         elseif iter_OD == n_OD
%             S13_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+2}:end) == 0;
%         else
%             S13_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+2}:idxEndCol{1, n_time_inc_start_ADMM+2}) == 0;
%             S13_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+2}:idxEndCol{1, n_time_inc_start_ADMM+2}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+2} = idxEndCol{1, n_time_inc_start_ADMM+2} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+3}>0
%         idxEndCol{1, n_time_inc_start_ADMM+3} = idxEndCol{1, n_time_inc_start_ADMM+3} + nDriverTemp{1, n_time_inc_start_ADMM+3};
%         if iter_OD==1
%             S14_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+3}) == 0;
%         elseif iter_OD == n_OD
%             S14_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+3}:end) == 0;
%         else
%             S14_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+3}:idxEndCol{1, n_time_inc_start_ADMM+3}) == 0;
%             S14_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+3}:idxEndCol{1, n_time_inc_start_ADMM+3}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+3} = idxEndCol{1, n_time_inc_start_ADMM+3} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+4}>0
%         idxEndCol{1, n_time_inc_start_ADMM+4} = idxEndCol{1, n_time_inc_start_ADMM+4} + nDriverTemp{1, n_time_inc_start_ADMM+4};
%         if iter_OD==1
%             S15_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+4}) == 0;
%         elseif iter_OD == n_OD
%             S15_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+4}:end) == 0;
%         else
%             S15_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+4}:idxEndCol{1, n_time_inc_start_ADMM+4}) == 0;
%             S15_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+4}:idxEndCol{1, n_time_inc_start_ADMM+4}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+4} = idxEndCol{1, n_time_inc_start_ADMM+4} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+5}>0
%         idxEndCol{1, n_time_inc_start_ADMM+5} = idxEndCol{1, n_time_inc_start_ADMM+5} + nDriverTemp{1, n_time_inc_start_ADMM+5};
%         if iter_OD==1
%             S16_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+5}) == 0;
%         elseif iter_OD == n_OD
%             S16_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+5}:end) == 0;
%         else
%             S16_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+5}:idxEndCol{1, n_time_inc_start_ADMM+5}) == 0;
%             S16_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+5}:idxEndCol{1, n_time_inc_start_ADMM+5}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+5} = idxEndCol{1, n_time_inc_start_ADMM+5} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+6}>0
%         idxEndCol{1, n_time_inc_start_ADMM+6} = idxEndCol{1, n_time_inc_start_ADMM+6} + nDriverTemp{1, n_time_inc_start_ADMM+6};
%         if iter_OD==1
%             S17_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+6}) == 0;
%         elseif iter_OD == n_OD
%             S17_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+6}:end) == 0;
%         else
%             S17_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+6}:idxEndCol{1, n_time_inc_start_ADMM+6}) == 0;
%             S17_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+6}:idxEndCol{1, n_time_inc_start_ADMM+6}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+6} = idxEndCol{1, n_time_inc_start_ADMM+6} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+7}>0
%         idxEndCol{1, n_time_inc_start_ADMM+7} = idxEndCol{1, n_time_inc_start_ADMM+7} + nDriverTemp{1, n_time_inc_start_ADMM+7};
%         if iter_OD==1
%             S18_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+7}) == 0;
%         elseif iter_OD == n_OD
%             S18_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+7}:end) == 0;
%         else
%             S18_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+7}:idxEndCol{1, n_time_inc_start_ADMM+7}) == 0;
%             S18_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+7}:idxEndCol{1, n_time_inc_start_ADMM+7}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+7} = idxEndCol{1, n_time_inc_start_ADMM+7} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+8}>0
%         idxEndCol{1, n_time_inc_start_ADMM+8} = idxEndCol{1, n_time_inc_start_ADMM+8} + nDriverTemp{1, n_time_inc_start_ADMM+8};
%         if iter_OD==1
%             S19_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+8}) == 0;
%         elseif iter_OD == n_OD
%             S19_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+8}:end) == 0;
%         else
%             S19_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+8}:idxEndCol{1, n_time_inc_start_ADMM+8}) == 0;
%             S19_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+8}:idxEndCol{1, n_time_inc_start_ADMM+8}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+8} = idxEndCol{1, n_time_inc_start_ADMM+8} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+9}>0
%         idxEndCol{1, n_time_inc_start_ADMM+9} = idxEndCol{1, n_time_inc_start_ADMM+9} + nDriverTemp{1, n_time_inc_start_ADMM+9};
%         if iter_OD==1
%             S110_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+9}) == 0;
%         elseif iter_OD == n_OD
%             S110_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+9}:end) == 0;
%         else
%             S110_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+9}:idxEndCol{1, n_time_inc_start_ADMM+9}) == 0;
%             S110_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+9}:idxEndCol{1, n_time_inc_start_ADMM+9}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+9} = idxEndCol{1, n_time_inc_start_ADMM+9} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+10}>0
%         idxEndCol{1, n_time_inc_start_ADMM+10} = idxEndCol{1, n_time_inc_start_ADMM+10} + nDriverTemp{1, n_time_inc_start_ADMM+10};
%         if iter_OD==1
%             S111_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+10}) == 0;
%         elseif iter_OD == n_OD
%             S111_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+10}:end) == 0;
%         else
%             S111_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+10}:idxEndCol{1, n_time_inc_start_ADMM+10}) == 0;
%             S111_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+10}:idxEndCol{1, n_time_inc_start_ADMM+10}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+10} = idxEndCol{1, n_time_inc_start_ADMM+10} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_ADMM+11}>0
%         idxEndCol{1, n_time_inc_start_ADMM+11} = idxEndCol{1, n_time_inc_start_ADMM+11} + nDriverTemp{1, n_time_inc_start_ADMM+11};
%         if iter_OD==1
%             S112_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+11}) == 0;
%         elseif iter_OD == n_OD
%             S112_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+11}:end) == 0;
%         else
%             S112_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+11}:idxEndCol{1, n_time_inc_start_ADMM+11}) == 0;
%             S112_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+11}:idxEndCol{1, n_time_inc_start_ADMM+11}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_ADMM+11} = idxEndCol{1, n_time_inc_start_ADMM+11} + 1;
%     end
%     idxEndRow1 = idxEndRow1 + nPathTemp;
% end
cvx_end
S_binary_full = cell(n_companies_ADMM, n_time);
S_binary_full{1, n_time_inc_start_ADMM+0} = full(S11_binary);
S_binary_full{1, n_time_inc_start_ADMM+1} = full(S12_binary);
S_binary_full{1, n_time_inc_start_ADMM+2} = full(S13_binary);
S_binary_full{1, n_time_inc_start_ADMM+3} = full(S14_binary);
S_binary_full{1, n_time_inc_start_ADMM+4} = full(S15_binary);
S_binary_full{1, n_time_inc_start_ADMM+5} = full(S16_binary);
S_binary_full{1, n_time_inc_start_ADMM+6} = full(S17_binary);
S_binary_full{1, n_time_inc_start_ADMM+7} = full(S18_binary);
S_binary_full{1, n_time_inc_start_ADMM+8} = full(S19_binary);
S_binary_full{1, n_time_inc_start_ADMM+9} = full(S110_binary);
S_binary_full{1, n_time_inc_start_ADMM+10} = full(S111_binary);
S_binary_full{1, n_time_inc_start_ADMM+11} = full(S112_binary);

S_check_response = cell(n_companies_ADMM, n_time);
for time_idx=0:11
    % All the values are 0 or 1 but some 0 and 1 values have different
    % decimals in high precision. Something like 1.00000001. To prevent
    % issues, we round all the numbers
    S_sol_full = round(S_binary_full{1, n_time_inc_start_ADMM+time_idx});
%         [row_S_sorted, col_S_sorted] = find(S_sol_full==1);
%         [out_temp, idx_S_sorted] = sort(row_S_sorted);
%         S_sorted = S_sol_full(:, idx_S_sorted);
%         S_check_response{1, n_time_inc_start_ADMM + time_idx} = S_sorted;
    S_check_response{1, n_time_inc_start_ADMM + time_idx} = transpose(sortrows(transpose(round(S_sol_full)),'descend'));
end

totalTimeLP = toc(time_LP_start);

%% Compute total travel time after binarization
tt = @(x, v, tt0, w) (x+v)*tt0 + 0.15*tt0*((x+v)^5)/(w^4);
omega_sol1 =  ...
    R_matrix(:, (n_time_inc_start_ADMM-1+0)*n_path+1:((n_time_inc_start_ADMM+0)*n_path))*S_check_response{1, n_time_inc_start_ADMM+0}*ones_S{1, n_time_inc_start_ADMM+0} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+1)*n_path+1:((n_time_inc_start_ADMM+1)*n_path))*S_check_response{1, n_time_inc_start_ADMM+1}*ones_S{1, n_time_inc_start_ADMM+1} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+2)*n_path+1:((n_time_inc_start_ADMM+2)*n_path))*S_check_response{1, n_time_inc_start_ADMM+2}*ones_S{1, n_time_inc_start_ADMM+2} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+3)*n_path+1:((n_time_inc_start_ADMM+3)*n_path))*S_check_response{1, n_time_inc_start_ADMM+3}*ones_S{1, n_time_inc_start_ADMM+3} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+4)*n_path+1:((n_time_inc_start_ADMM+4)*n_path))*S_check_response{1, n_time_inc_start_ADMM+4}*ones_S{1, n_time_inc_start_ADMM+4} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+5)*n_path+1:((n_time_inc_start_ADMM+5)*n_path))*S_check_response{1, n_time_inc_start_ADMM+5}*ones_S{1, n_time_inc_start_ADMM+5} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+6)*n_path+1:((n_time_inc_start_ADMM+6)*n_path))*S_check_response{1, n_time_inc_start_ADMM+6}*ones_S{1, n_time_inc_start_ADMM+6} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+7)*n_path+1:((n_time_inc_start_ADMM+7)*n_path))*S_check_response{1, n_time_inc_start_ADMM+7}*ones_S{1, n_time_inc_start_ADMM+7} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+8)*n_path+1:((n_time_inc_start_ADMM+8)*n_path))*S_check_response{1, n_time_inc_start_ADMM+8}*ones_S{1, n_time_inc_start_ADMM+8} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+9)*n_path+1:((n_time_inc_start_ADMM+9)*n_path))*S_check_response{1, n_time_inc_start_ADMM+9}*ones_S{1, n_time_inc_start_ADMM+9} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+10)*n_path+1:((n_time_inc_start_ADMM+10)*n_path))*S_check_response{1, n_time_inc_start_ADMM+10}*ones_S{1, n_time_inc_start_ADMM+10} +...
    R_matrix(:, (n_time_inc_start_ADMM-1+11)*n_path+1:((n_time_inc_start_ADMM+11)*n_path))*S_check_response{1, n_time_inc_start_ADMM+11}*ones_S{1, n_time_inc_start_ADMM+11} +...
    0;

tt_obj = zeros(n_link*n_time, 1);
for iter_omega = 1:n_link*n_time
    tt0 = tt0_array(link_loc(mod(iter_omega-1, size(tt0_array, 1)) + 1, 1), 3);
    w = w_array(link_loc(mod(iter_omega-1, size(w_array, 1)) + 1, 1), 3);
    v_dynamic = ...
        omega_sol1(iter_omega, 1) +  ...
        0;
    v_fixed = v_tilda_baseline(iter_omega, 1);
    tt_temp = tt(v_dynamic, v_fixed, tt0, w);
    tt_obj(iter_omega, 1) = tt_temp;
    if mod(iter_omega-1, n_link)==n_link-1
        fprintf('Total travel time at time %i: %.6f hours\n', iter_omega/n_link, sum(tt_obj((iter_omega/n_link-1)*n_link+1:iter_omega))/60)
    end
end
tt_obj_total = sum(tt_obj)/60;
fprintf('Total travel time after incentivization: %.6f hours\n', tt_obj_total)


tt_single = @(x, v, tt0, w) tt0 + 0.15*tt0*((x+v)^4)/(w^4);
tt_obj_user_NoInc = zeros(n_link*n_time, 1);
tt_obj_user_ADMM = zeros(n_link*n_time, 1);
tt_obj = zeros(n_link*n_time, 1);
tt_obj_NoInc = zeros(n_link*n_time, 1);
for iter_omega = 1:n_link*n_time
    tt0 = tt0_array(link_loc(mod(iter_omega-1, size(tt0_array, 1)) + 1, 1), 3);
    w = w_array(link_loc(mod(iter_omega-1, size(w_array, 1)) + 1, 1), 3);

    v_fixed = v_tilda_baseline(iter_omega, 1);

    v_user_NoInc = v_dynamic_array_NoInc_allDet(iter_omega, 1);
    tt_temp_NoInc = tt(v_user_NoInc, v_fixed, tt0, w);
    tt_obj_NoInc(iter_omega, 1) = tt_temp_NoInc;
    if mod(iter_omega-1, n_link)==n_link-1
        fprintf('Total travel time at time %i before incentivization: %.6f hours\n', iter_omega/n_link, sum(tt_obj_NoInc((iter_omega/n_link-1)*n_link+1:iter_omega))/60)
    end
    tt_temp_single_NoInc = tt_single(v_user_NoInc, v_fixed, tt0, w);
    tt_obj_user_NoInc(iter_omega, 1) = tt_temp_single_NoInc * v_user_NoInc;

    v_dynamic = omega_sol1(iter_omega, 1);
    tt_temp = tt(v_dynamic, v_fixed, tt0, w);
    tt_obj(iter_omega, 1) = tt_temp;
    if mod(iter_omega-1, n_link)==n_link-1
        fprintf('Total travel time at time %i: %.6f hours\n', iter_omega/n_link, sum(tt_obj((iter_omega/n_link-1)*n_link+1:iter_omega))/60)
    end
    tt_temp_single_ADMM = tt_single(v_dynamic, v_fixed, tt0, w);
    tt_obj_user_ADMM(iter_omega, 1) = tt_temp_single_ADMM * v_dynamic;
end
tt_obj_total_NoInc = sum(tt_obj_NoInc)/60;
fprintf('Total travel time before incentivization: %.6f hours\n', tt_obj_total_NoInc)
tt_obj_total = sum(tt_obj)/60;
fprintf('Total travel time after incentivization: %.6f hours\n', tt_obj_total)
tt_obj_total_NoInc_7_8 = sum(tt_obj_NoInc(n_link*n_time_inc_start_ADMM+1:n_link*n_time_inc_end_ADMM))/60;
fprintf('Travel time between 7AM and 8AM before incentivization: %.6f hours\n', tt_obj_total_NoInc_7_8)
tt_obj_7_8 = sum(tt_obj(n_link*n_time_inc_start_ADMM+1:n_link*n_time_inc_end_ADMM))/60;
fprintf('Travel time between 7AM and 8AM after incentivization: %.6f hours\n', tt_obj_7_8)
tt_obj_total_NoInc_7_830 = sum(tt_obj_NoInc(n_link*12+1:n_link*30))/60;
fprintf('Travel time between 7AM and 8:30AM before incentivization: %.6f hours\n', tt_obj_total_NoInc_7_830)
tt_obj_7_830 = sum(tt_obj(n_link*12+1:n_link*30))/60;
fprintf('Travel time between 7AM and 8:30AM after incentivization: %.6f hours\n', tt_obj_7_830)
tt_obj_user_NoInc = sum(tt_obj_user_NoInc)/60;
fprintf('User total travel time before incentivization: %.6f hours\n', tt_obj_user_NoInc)
tt_obj_user_ADMM = sum(tt_obj_user_ADMM)/60;
fprintf('User total travel time after incentivization: %.6f hours\n', tt_obj_user_ADMM)
%% Gaps of constraints
S_binary_full_blkdiag = cell(n_companies_ADMM, 1);
% S_binary_full_blkdiag{1, 1} = blkdiag(S_check_response{1, :});
S_binary_full_blkdiag{1, 1} = transpose(sortrows(transpose(blkdiag(S_check_response{1, :})),'descend'));

const11_gap =  [zeros((n_time_inc_start_ADMM-1)*n_OD, 1);...
    D((n_time_inc_start_ADMM-1)*n_OD+1:n_time_inc_end_ADMM*n_OD, (n_time_inc_start_ADMM-1)*n_path+1:n_time_inc_end_ADMM*n_path)*S_binary_full_blkdiag{1, 1}*...
    ones(n_driver_company{1}, 1); zeros((n_time-n_time_inc_end_ADMM)*n_OD, 1)] - q_company{1};
sum_const11_gap = sum(abs(const11_gap));
fprintf('Gap of number of drivers of ODs constraint 1: %.6f\n', sum_const11_gap)
fprintf("\n")
const21_gap = S_binary_full_blkdiag{1, 1}'*ones(n_path*(n_time_inc_end_ADMM-n_time_inc_start_ADMM+1), 1) - ones(n_driver_company{1}, 1);
sum_const21_gap = sum(abs(const21_gap));
fprintf('Gap of single path assignment constraint 1: %.6f\n', sum_const21_gap)
fprintf("\n")


% RHS of baseline
RHS_baseline = cell(n_companies_ADMM, 1);
for iter_company=1:n_companies_ADMM
    RHS_baseline{iter_company} = B{iter_company} * etaDet;
end
nDeviatedCompany = cell(n_companies_ADMM, 1);
nDeviatedCompany(:) = {0};
nDeviatedCompanyOD = cell(n_companies_ADMM, n_OD);
nDeviatedCompanyOD(:) = {0};
deviatedOD = cell(n_companies_ADMM, 1);
deviatedOD(:) = {[]};
for iter_company=1:n_companies_ADMM
    idxEndRow1 = 0;
    idxStartCol = 1;
    idxEndCol = 0;
    for iter_time=n_time_inc_start_ADMM:n_time_inc_end_ADMM
        for iter_OD=1:n_OD
            nPathTemp = num_path_v(iter_OD, 1);
            nDriverTemp  = q_company_time{iter_company, iter_time}(iter_OD, 1);
            if nDriverTemp>0
                idxEndCol = idxEndCol + nDriverTemp;
                nDeviatedTemp = 0;
                for iter_driver=1:nDriverTemp
                    assert(sum(S_binary_full_blkdiag{iter_company, 1}((idxEndRow1+1):(idxEndRow1+nPathTemp), idxStartCol+(iter_driver-1))) == 1);
                    if ((S_binary_full_blkdiag{iter_company, 1}(:, idxStartCol+(iter_driver-1)))'*delta_pDet(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM) - RHS_baseline{iter_company}(idxStartCol+(iter_driver-1), 1))>0
                        nDeviatedTemp = nDeviatedTemp + 1;
                    end
                end
                if nDeviatedTemp>0
                    deviatedOD{iter_company, 1} = [deviatedOD{iter_company, 1}; iter_OD];
                    nDeviatedCompany{iter_company, 1} = nDeviatedCompany{iter_company, 1} + nDeviatedTemp;
                    nDeviatedCompanyOD{iter_company, iter_OD} = nDeviatedCompanyOD{iter_company, iter_OD} + nDeviatedTemp;
                    fprintf('%i drivers of company %i deviated at time %i from OD %i.\n', nDeviatedTemp, iter_company, iter_time, iter_OD)
                end
                idxStartCol = idxEndCol + 1;
            end
            idxEndRow1 = idxEndRow1 + nPathTemp;
        end
    end
    fprintf('# of deviated drivers of company %i: %i drivers\n', iter_company, nDeviatedCompany{iter_company, 1})
    fprintf('ODs that drivers of company %i are deviated from:\n', iter_company)
    fprintf('%i \n', sort(unique(deviatedOD{iter_company, 1})))
end
for iter_company=1:n_companies_ADMM
    for iter_OD=1:n_OD
        if nDeviatedCompanyOD{iter_company, iter_OD}>0
            fprintf('# of deviated drivers of company %i in OD %i: %i drivers\n', iter_company, iter_OD, nDeviatedCompanyOD{iter_company, iter_OD})
        end
    end
    fprintf('\n')
end
fprintf('Assertion of order of drivers in assignment is PASSED.\n')

gap_baseline1 = S_binary_full_blkdiag{1, 1}'*delta_pDet(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM) - RHS_baseline{1};
assert(size(gap_baseline1(gap_baseline1<-0.001), 1)==0) % Check being above min tt
fprintf('Assertion of positive deviation from min tt is PASSED.\n')
nDeviated1 = size(gap_baseline1(gap_baseline1>0), 1);
sum_gap_baseline1 = sum(gap_baseline1);
fprintf('Deviation from minimum travel time constraint 1: %.6f hours\n', sum_gap_baseline1/60)
fprintf('Total deviation from minimum travel time of companies: %.6f hours\n', (sum_gap_baseline1 + ...
    0)/60)
fprintf('# of deviated drivers of company 1: %i drivers\n', nDeviated1)
fprintf('Total # of deviated drivers of companies: %i drivers\n', nDeviated1+...
    0)
fprintf('Avrege deviation of deviated drivers of company 1: %.6f hours\n', sum_gap_baseline1/60/nDeviated1)
fprintf('Avrege deviation of deviated drivers of companies: %.6f hours\n', (sum_gap_baseline1+...
    0)/60/(nDeviated1+...
    0))
fprintf("\n")
const31_gap = S_binary_full_blkdiag{1, 1}'*delta_pDet(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM) - RHS{1};
sum_abs_const31_gap = sum(abs(const31_gap));
const31_gap_positive = const31_gap(const31_gap>0);
const31_gap_negative = const31_gap(const31_gap<0);
sum_pos_const31_gap = sum((const31_gap_positive));
sum_neg_const31_gap = sum((const31_gap_negative));
fprintf('Absoulte gap of minimum travel time constraint 1: %.6f hours\n', sum_abs_const31_gap/60)
fprintf('Positive deviation from minimum travel time constraint 1: %.6f hours\n', sum_pos_const31_gap/60)
fprintf('Negative deviation from minimum travel time constraint 1: %.6f hours\n', sum_neg_const31_gap/60)
fprintf("\n")
gap_cost1 = c(1) - alpha(1).*(delta_pDet(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM)'*S_binary_full_blkdiag{1, 1}*ones(n_driver_company{1}, 1)-gamma(1));
cost_1 = alpha(1).*(delta_pDet(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM)'*S_binary_full_blkdiag{1, 1}*ones(n_driver_company{1}, 1)-gamma(1));
fprintf('c(1) (LP cost variable of company 1): %.6f\n', c(1))
fprintf('Cost of company 1 (Result of LP): %.6f\n', cost_1)
fprintf('Gap of cost constraint 1 (LP): %.6f\n', gap_cost1)
fprintf('Total budget: %.4f\n', budget)
fprintf('Utalized budget of ADMM: %.6f\n', c_tilda(1, 1) +...
    0)
fprintf('Utalized budget of LP cost variables: %.6f\n', c(1) + ...
    0)
fprintf('Utalized budget from LP results: %.6f\n', cost_1 + ...
    0)
fprintf('Remained budget of LP: %.6f\n', (budget - (c(1) + ...
    0)))
fprintf('Remained budget of LP results: %.6f\n', (budget - (cost_1 + ...
    0)))
fprintf('Total # of drivers: %.6f\n', sum(q))
fprintf('Total # of nonuser drivers: %.6f\n', sum(q_nonuser))
fprintf('Total # of user drivers: %.6f\n', sum(q_user))
fprintf('# of drivers working at company 1: %.6f\n', n_driver_company{1})
fprintf('Cost per driver of company 1: %.6f\n', cost_1/n_driver_company{1})
fprintf('Cost per deviated drivers of company 1: %.6f\n', cost_1/nDeviated1)
fprintf('Cost per driver of companies: %.6f\n', (cost_1 + ...
    0)/(n_driver_company{1}+...
    0))
fprintf("\n")

fprintf('Travel time between 7AM and 8:30AM before incentivization: %.6f hours\n', tt_obj_total_NoInc_7_830)
fprintf('Travel time between 7AM and 8:30AM after incentivization: %.6f hours\n', tt_obj_7_830)
fprintf('Travel time between 7AM and 8:30AM improvement: %.2f\n', (tt_obj_total_NoInc_7_830-tt_obj_7_830)/tt_obj_total_NoInc_7_830*100)
%% Computation times
totalTimeRun = toc(startTime0);
fprintf('Total run time LP: %.6f minutes\n', totalTimeLP/60)
fprintf('Total run time: %.6f minutes\n', totalTimeRun/60)

%% Keep record of runtime
% Define the file name
fileNameRuntime = fullfile(folderRun, 'runtimes_ILP.csv');
% Check if the file exists
if isfile(fileNameRuntime)
    % File exists, load existing data
    existingRuntime = readmatrix(fileNameRuntime);
    % Append new data
    updatedRuntime = [existingRuntime; MIPGap, totalTimeLP/60, totalTimeRun/60];
else
    % File does not exist, initialize with newData
    updatedRuntime = [MIPGap, totalTimeLP/60, totalTimeRun/60];
end

% Save updated data to CSV
writematrix(updatedRuntime, fileNameRuntime);

%% Save arrays
fileName2 = fullfile(folderRun,...
                strcat('result_MIPGap', num2str(MIPGap), '_ILP.mat'));
save(fileName2);

% Beep sound
sound(sin(1:3000));
diary off; % Stop logging
% Python script location
pythonScript = fullfile('../lib', 'convert_to_pdf.py');
% Call the Python script
% system(sprintf('python "%s"', pythonScript));
system(sprintf('python3 "%s" "%s"', pythonScript, log_file));
