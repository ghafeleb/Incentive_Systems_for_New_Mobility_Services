function compareCosts_realCost_initAll2(iterRun, nonuser_perc_prob,...
    budget, n_companies_solving_algo, n_companies_cost, n_iter_ADMM,...
    n_time, n_time_inc_start, n_time_inc_end, ...
    fairness, VOT, seed_solving_algo, seedData, rho, step_size, ...
    region_, setting_region, MIPGap)
%% Initialization 
fprintf('n_companies_cost: %i\n', n_companies_cost)
iterRun = iterRun+1;
nonuser_prob = nonuser_perc_prob/100;
% Split the string at each underscore
strArray = split(fairness, '_');
% Convert the split string array to double precision numbers
prob_fairness = transpose(str2double(strArray));  % [x1.0, x1.1, x1.5, x2.0, x2.5]

LogicalStr = {'F', 'T'};

% Percentace of nonuser drivers in each time interval
nonuser_perc_ADMM = repmat(nonuser_prob, n_time, 1);
% Initialize the decision matrix S with the baseline
initializeSBaseline = true;
rng(seed_solving_algo) % Specification of seed for the random number generator (rng)

assert(sum(prob_fairness./100)==1) % Sum of probs in dist should be 1
%% Load Incentivization Results
setting_output = sprintf('%.0f', nonuser_perc_ADMM(1)*100);
inputFolder0 = fullfile('../data', region_, setting_region);

if n_iter_ADMM < 0
    if n_iter_ADMM == -1
        solver_name = "Gurobi"
    elseif n_iter_ADMM == -2
        solver_name = "Mosek"
    elseif n_iter_ADMM == -3
        solver_name = "GLPK"
    end
    folderRun = fullfile(inputFolder0, ...
                        strcat(solver_name,'_new_Det_initAll2_MultT', ...
                                '_b', num2str(budget), ...
                                '_sD', num2str(seedData), ...
                                '_sS', num2str(seed_solving_algo), ...
                                '_VOT', num2str(VOT), ...
                                '_nC', num2str(1), ...
                                '_f', fairness, ...
                                '_percNonU', num2str(setting_output), ...
                                '_nTIS', num2str(n_time_inc_start), ...
                                '_nTIE', num2str(n_time_inc_end), ...
                                '_ss', num2str(step_size), ...
                                '_itN', num2str(iterRun)));
%     fileIncentivized = fullfile(folderRun, strcat(solver_name, ...
%                                     '_solver_result.mat'));
    fileIncentivized = fullfile(folderRun, strcat(solver_name, ...
                            '_MIPGap', num2str(MIPGap), '_solver_result.mat'));
    load(fileIncentivized, 'S_binary_full*', 'num_path_v', 'gamma', 'alpha');
                                
    fileSolverData = fullfile(folderRun,...
        strcat('result_', num2str(iterRun), '_ADMMDataInit.mat'));
    load(fileSolverData, 'etaDet', 'B', 'n_driver_company');
    
else
    folderStr = 'Det_initAll2_MultT';
    folderRun = fullfile(inputFolder0, ...
                    strcat(folderStr, ...
                    '_b', num2str(budget), ...
                    '_sD', num2str(seedData), ...
                    '_sA', num2str(seed_solving_algo), ...
                    '_r', num2str(rho), ...
                    '_it', num2str(n_iter_ADMM),...
                    '_VOT', num2str(VOT), ...
                    '_nC', num2str(n_companies_solving_algo), ...
                    '_f', fairness, ...
                    '_initSB_', LogicalStr{initializeSBaseline + 1}, ...
                    '_percNonU', setting_output, ...
                    '_nTIS', num2str(n_time_inc_start), ...
                    '_nTIE', num2str(n_time_inc_end), ...
                    '_ss', num2str(step_size), ...
                    '_itN', num2str(iterRun)));
    fileIncentivized = fullfile(folderRun, ...
        strcat('result_MIPGap', num2str(MIPGap), '_ILP.mat'));
%                                 strcat('result_',...
%                                 num2str(iterRun), ...
%                                 '.mat'));
    load(fileIncentivized, 'S_binary_full*', 'num_path_v', 'gamma', ...
        'etaDet', 'alpha', 'n_driver_company', 'B');
end
folderCost = fullfile(folderRun, strcat('cost_MIPGap', num2str(MIPGap)));
log_file = fullfile(folderCost, strcat('compare_costs_nC', num2str(n_companies_cost), '.txt'));
diary(log_file); % Start logging to a file named 'outputLog.txt'

%% Arrays of travel time Deterministic
% TT of each path. Size: (max_num_path*n_time)x(n_OD)
date_tt = "2018-05-01";
tt_path_incentivized = readmatrix(fullfile(folderCost, strcat('tt', date_tt, '.csv')))/60; % Seconds to minutes

max_num_path = max(num_path_v);
n_path = sum(num_path_v);
n_OD = size(tt_path_incentivized, 2);
eta_incentivized = zeros(n_OD*n_time, 1);
delta_p_incentivized = zeros(n_path*n_time, 1);
I_minTT_initAll2 = cell(n_time, 1);
I_minTT_initAll2(:) = {zeros(n_OD, 1)};
for iter_time=1:n_time
    idxTemp = 0; % Starting idx of path
    % Create delta_p_incentivized
    index_0 = tt_path_incentivized(max_num_path*(iter_time-1)+1, :)==0;
    tt_path_incentivized(max_num_path*(iter_time-1)+1, index_0) = 0.0001;
    temp_delta = tt_path_incentivized(max_num_path*(iter_time-1)+1:max_num_path*(iter_time), :);
    temp_delta = temp_delta(:);
    temp_delta(temp_delta==0) = [];
    delta_p_incentivized(n_path*(iter_time-1)+1:n_path*(iter_time)) = temp_delta;
    
    % Create eta_incentivized
    row_index = max_num_path*(iter_time-1)+1:max_num_path*(iter_time);
    for iter_OD=1:n_OD
        tempMinTT = tt_path_incentivized(row_index, iter_OD);
        tempMinTT(tempMinTT==0) = [];
        [tempMinTT, tempMinTTIdx] = min(tempMinTT);
        eta_incentivized((iter_time-1)*n_OD+iter_OD, 1) = tempMinTT;
        I_minTT_initAll2{iter_time, 1}(iter_OD, 1) = idxTemp + tempMinTTIdx;
        idxTemp = idxTemp + num_path_v(iter_OD, 1);
    end
end
delta_p_incentivized(delta_p_incentivized==0.0001) = 0;
eta_incentivized(eta_incentivized==0.0001) = 0;
clearvars index_0 temp_delta row_index tempMinTT iter_time iter_OD

%% Compute real cost v1
cost_v1 = zeros(n_companies_cost, 1);
ttDecrease = zeros(n_companies_cost, 1);
if n_companies_cost==n_companies_solving_algo
    for iter_company=1:n_companies_cost
        ttDecrease(iter_company, 1) = delta_p_incentivized(n_path*(n_time_inc_start-1)+1:n_path*n_time_inc_end)'*blkdiag(S_binary_full{iter_company, :})*ones(n_driver_company{iter_company}, 1)-gamma(iter_company);
        cost_v1(iter_company, 1) = alpha(iter_company).*(delta_p_incentivized(n_path*(n_time_inc_start-1)+1:n_path*n_time_inc_end)'*blkdiag(S_binary_full{iter_company, :})*ones(n_driver_company{iter_company}, 1)-gamma(iter_company));
        fprintf('Cost of company %i V1-1: %.6f\n', iter_company, cost_v1(iter_company, 1))
        fprintf('tt change of company %i (hour): %.6f\n', iter_company, ttDecrease(iter_company, 1)/60)
    end
elseif n_companies_solving_algo==1
    rng(seedData)
    S_blkdiag = blkdiag(S_binary_full{1, :});
    alpha_new = ones(n_companies_cost, 1)*VOT; % Value of time
    gamma_new = zeros(n_companies_cost, 1);
    n_driver_company_new = cell(n_companies_cost, 1);
    n_t = size(S_blkdiag, 2);
    fprintf('Total # of users: %i\n', n_t)
    n_c_floor = floor(1/n_companies_cost*n_t);
    n_c_round = n_c_floor*n_companies_cost;
    n_c_mod = mod(n_t, n_c_round);
    remained_users = 1:n_t;
    B = B{1};
    total_cost = 0;
    total_tt = 0;
    for iter_company=1:n_companies_cost
        if iter_company<=n_c_mod
            n_c_temp = n_c_floor + 1;
        else
            n_c_temp = n_c_floor;
        end
        n_driver_company_new{iter_company} = n_c_temp;
        fprintf('# of drivers of company %i: %i\n', iter_company, n_c_temp)
        selected_users_temp = randsample(remained_users, n_c_temp);
        remained_users = setdiff(remained_users, selected_users_temp);
        gamma_new(iter_company) = (B(selected_users_temp, :)*etaDet)'*ones(n_c_temp, 1);
        fprintf('gamma_new(%i) V1-2: %.6f\n', iter_company, gamma_new(iter_company))
        ttDecrease(iter_company, 1) = delta_p_incentivized(n_path*(n_time_inc_start-1)+1:n_path*n_time_inc_end)'*S_blkdiag(:, selected_users_temp)*ones(n_driver_company_new{iter_company}, 1)-gamma_new(iter_company);
        cost_v1(iter_company, 1) = alpha_new(iter_company).*(ttDecrease(iter_company, 1));
        if cost_v1(iter_company, 1)~=0
            fprintf('Cost of company %i V1-2: %.6f\n', iter_company, cost_v1(iter_company, 1))
        end
        if ttDecrease(iter_company, 1)~=0
            fprintf('Travel time change of company %i (hour): %.6f\n', iter_company, ttDecrease(iter_company, 1)/60)
        end
        if cost_v1(iter_company, 1)>0
            total_cost = total_cost + cost_v1(iter_company, 1);
        end
        if ttDecrease(iter_company, 1)>0
            total_tt = total_tt + ttDecrease(iter_company, 1);
        end
        total_tt = total_tt + ttDecrease(iter_company, 1);

    end
    fprintf('Total cost: %.6f\n', total_cost)
    fprintf('Total tt increase V1 (hour): %.6f\n', total_tt/60)
end
    
diary off; % Stop logging
% Python script location
pythonScript = fullfile('../lib', 'convert_to_pdf.py');
% Call the Python script
% system(sprintf('python "%s"', pythonScript));
system(sprintf('python3 "%s" "%s"', pythonScript, log_file));
