function compareCosts_realCost_initAll2_allInOne(iterRun, nonuser_perc_prob,...
    budget, n_companies_solving_algo, n_iter_ADMM,...
    min_n_companies, max_n_companies, step_n_companies, factor_n_companies, n_sample, ...
    n_time, n_time_inc_start, n_time_inc_end, ...
    fairness, VOT, seed_solving_algo, seedData, rho, step_size, ...
    region_, setting_region, MIPGap)
%% Initialization 
% fprintf('n_companies_cost: %i\n', n_companies_cost)
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
                            
    fileIncentivized = fullfile(folderRun, strcat(solver_name, ...
                            '_MIPGap', num2str(MIPGap), '_solver_result.mat'));
%     fileIncentivized = fullfile(folderRun, strcat(solver_name, ...
%                                     '_solver_result.mat'));
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
        '_nTIS', num2str(n_time_inc_start), '_nTIE', num2str(n_time_inc_end), ...
        '_ss', num2str(step_size), ...
        '_itN', num2str(iterRun)));
    fileIncentivized = fullfile(folderRun, ...        
        strcat('result_MIPGap', num2str(MIPGap), '_ILP.mat'));
%         strcat('result_',...
%         num2str(iterRun), ...
%         '.mat'));
    load(fileIncentivized, 'S_binary_full*', 'num_path_v', 'gamma', ...
        'etaDet', 'alpha', 'n_driver_company', 'B');
end

if factor_n_companies ~= 0
    fileName_suffix = strcat('_factor', num2str(factor_n_companies));
else
    fileName_suffix = strcat('_min', ...
                            num2str(min_n_companies), '_step',...
                            num2str(step_n_companies), '_max',...
                            num2str(max_n_companies), '_nSample',...
                            num2str(n_sample));
end
folderCost = fullfile(folderRun, strcat('cost_MIPGap', num2str(MIPGap)));
log_file = fullfile(folderCost, strcat('compare_costs_nC', fileName_suffix, '.txt'));
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
S_blkdiag = blkdiag(S_binary_full{1, :});
B_temp = B{1};
n_user_total = size(S_blkdiag, 2);
if factor_n_companies > 0
    n_companies_cost_array = [1];
    while  n_companies_cost_array(end)*factor_n_companies < n_user_total
        n_companies_cost_array = [n_companies_cost_array, ...
                        n_companies_cost_array(end)*factor_n_companies];
    end
    n_companies_cost_array = [n_companies_cost_array, n_user_total];
elseif factor_n_companies == -1
    n_companies_cost_array = [1,2,3,5];
else
    if min_n_companies==1
        n_companies_cost_array = [1, ...
                            step_n_companies:step_n_companies:max(max_n_companies, n_user_total)];
    else
        n_companies_cost_array = [1, ...
                            min_n_companies:step_n_companies:max(max_n_companies, n_user_total)];
    end 
    if n_companies_cost_array(end) ~= n_user_total
        n_companies_cost_array = [n_companies_cost_array, n_user_total];
    end
end
n_companies = size(n_companies_cost_array, 2);
% cost_summary: col1=cost, col2=tt change (hour), col3=company costs
cost_summary = cell(n_companies , 8); 
for n_companies_idx = 1:n_companies
    rng(seedData)
    n_companies_cost = n_companies_cost_array(1, n_companies_idx);
    cost_v1 = zeros(n_companies_cost, 1);
    ttDecrease = zeros(n_companies_cost, 1);
    total_cost_array = [];
    total_tt_array = [];
    alpha_new = ones(n_companies_cost, 1)*VOT; % Value of time
    for iter_sample =1:n_sample
        gamma_new = zeros(n_companies_cost, 1);
        n_driver_company_new = cell(n_companies_cost, 1);
%         fprintf('Total # of users: %i\n', n_user_total)
        n_user_per_company_floor = floor(n_user_total/n_companies_cost); % Number of drivers per company
        % Compute how many user drivers are remained if companies have
        % n_user_per_company_floor number of drivers
        n_user_total_round = n_user_per_company_floor*n_companies_cost; 
        n_user_mod = mod(n_user_total, n_user_total_round);
        remained_users = 1:n_user_total; % Remained users will be assigned later
        total_cost = 0;
        total_gain = 0;
        total_tt = 0;
        n_increased_tt_company = 0;
        n_decreased_tt_company = 0;
        for iter_company=1:n_companies_cost
            if iter_company<=n_user_mod
                n_user_company = n_user_per_company_floor + 1;
            else
                n_user_company = n_user_per_company_floor;
            end
            n_driver_company_new{iter_company} = n_user_company;
%             fprintf('# of drivers of company %i: %i\n', iter_company, n_user_company)
            selected_users_temp = randsample(remained_users, n_user_company);
            remained_users = setdiff(remained_users, selected_users_temp);
            gamma_new(iter_company) = (B_temp(selected_users_temp, :)*etaDet)'*ones(n_user_company, 1);
%             fprintf('gamma_new(%i) V1-2: %.6f\n', iter_company, gamma_new(iter_company))
            ttDecrease(iter_company, 1) = delta_p_incentivized(n_path*(n_time_inc_start-1)+1:n_path*n_time_inc_end)'*S_blkdiag(:, selected_users_temp)*ones(n_driver_company_new{iter_company}, 1)-gamma_new(iter_company);
            cost_v1(iter_company, 1) = alpha_new(iter_company).*(ttDecrease(iter_company, 1));
            if cost_v1(iter_company, 1)~=0
                a_pass = 0;
%                 fprintf('Cost of company %i V1-2: %.6f\n', iter_company, cost_v1(iter_company, 1))
            end
            if ttDecrease(iter_company, 1)~=0
                a_pass = 0;
%                 fprintf('Travel time change of company %i (hour): %.6f\n', iter_company, ttDecrease(iter_company, 1)/60)
            end
            if cost_v1(iter_company, 1)>0
                total_cost = total_cost + cost_v1(iter_company, 1);
                n_increased_tt_company = n_increased_tt_company + 1;
            else
                total_gain = total_gain + cost_v1(iter_company, 1);
                n_decreased_tt_company = n_decreased_tt_company + 1;
            end
            if ttDecrease(iter_company, 1)>0
                total_tt = total_tt + ttDecrease(iter_company, 1);
            end
            total_tt = total_tt + ttDecrease(iter_company, 1);

        end
%             fprintf('Total cost of %i companies: %.6f\n', n_companies_cost, total_cost)
        total_cost_array = [total_cost_array;total_cost];
        total_tt_array = [total_tt_array;total_tt];
    end
%         fprintf('Total tt increase V1 (hour): %.6f\n', total_tt/60)
    fprintf('Average cost of %i companies: %.6f\n', n_companies_cost, mean(total_cost_array))
    cost_summary{n_companies_idx, 1} = n_companies_cost;
    cost_summary{n_companies_idx, 2} = mean(total_cost_array);
    cost_summary{n_companies_idx, 3} = mean(total_tt_array)/60;
%         cost_summary{n_companies_cost, 1} = total_cost;
%         cost_summary{n_companies_cost, 2} = total_tt/60;
        cost_summary{n_companies_idx, 4} = cost_v1;
        cost_summary{n_companies_idx, 5} = ttDecrease;
        cost_summary{n_companies_idx, 6} = n_driver_company_new;
        cost_summary{n_companies_idx, 7} = n_increased_tt_company;
        cost_summary{n_companies_idx, 8} = n_decreased_tt_company;
        
%     end
end
avg_costs = cost_summary(:, 2);
avg_costs_array = cell2mat(avg_costs);
plot(avg_costs_array)

save(fullfile(folderCost, strcat('compare_costs_nC', fileName_suffix, '.mat')));

diary off; % Stop logging
% Python script location
pythonScript = fullfile('../lib', 'convert_to_pdf.py');
% Call the Python script
% system(sprintf('python "%s"', pythonScript));
system(sprintf('python3 "%s" "%s"', pythonScript, log_file));
