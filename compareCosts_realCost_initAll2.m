
function compareCosts_realCost_initAll2(iterRun, nonuser_perc_prob_ADMM0, budget, n_companies_cost, MaxIter2, n_time, n_time_inc_start_ADMM, n_time_inc_end_ADMM)
setting_region = '5_22_AVG5_th1_pad_MultipleTimes';
region_ = 'region_toy';
% fprintf('n_time: %i\n', n_time)
% fprintf('n_time_inc_start_ADMM: %i\n', n_time_inc_start_ADMM)
% fprintf('n_time_inc_end_ADMM: %i\n', n_time_inc_end_ADMM)
VOT = 2.63;
% fprintf('VOT: %.6f\n', VOT)
fprintf('n_companies_cost: %i\n', n_companies_cost)
% n_companies = 1; % # of companies
n_companies = 1; % # of companiesaa
% fprintf('n_companies: %i\n', n_companies)
% fprintf('nonuser_perc_prob_ADMM0: %i\n', nonuser_perc_prob_ADMM0)
step_size = 0.01;
% fprintf('step_size: %f\n', step_size)
iterRun = iterRun+1;
% fprintf('iterRun: %i\n', iterRun)
% fprintf(strcat('region_:', region_, '\n'))
nonuser_perc_prob_ADMM = nonuser_perc_prob_ADMM0/100;
prob_fairness = [0/100, 0/100, 0/100, 100/100, 0/100];
seedADMM = 2;
% fprintf('seedADMM: %i\n', seedADMM)
seedData = 2;
% fprintf('seedData: %i\n', seedData)
rho = 20;
% fprintf('rho: %i\n', rho)
% MaxIter2 = 2000;
% fprintf('MaxIter2: %i\n', MaxIter2)
LogicalStr = {'F', 'T'};
perc_companies = [0.5; 0.5]; % Percentage of drivers in each company
% fprintf('perc_companies: [')
% fprintf('%.2f  ', perc_companies)
% fprintf(']\n')
setting_perc_companies = strcat(sprintf('%.0f', perc_companies(1)*100), ...
    '_', sprintf('%.0f', perc_companies(2)*100));
initializeSBaseline = true;
% fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})

%% Initialization 1
RHSMultiplier = [1, 1.1, 1.5, 2, 2.5];
MaxIter2 = MaxIter2 + seedADMM; % # of iterations of ADMM
% Print norms every normWindow steps
printNorm = false;
% fprintf('printNorm: %s\n', LogicalStr{printNorm + 1})
normWindow = 1000;
% fprintf('normWindow: %i\n', normWindow)
% Save LP approximations of last iterations
saveLastIters = false;
% fprintf('saveLastIters: %s\n', LogicalStr{saveLastIters + 1})
% Percentace of nonuser drivers in each time interval
nonuser_perc_ADMM = repmat(nonuser_perc_prob_ADMM, n_time, 1);
% Initialize the decision matrix S with the baseline
initializeSBaseline = true;
% fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})
% seedADMM = 2
rng(seedADMM) % Specification of seed for the random number generator (rng)
% Distribution of driver's tt upperbound
% prob_fairness = [0/100, 0/100, 100/100, 0/100, 0/100] % [x1.0, x1.1, x1.5, x2.0, x2.5]
% fprintf('RHSMultiplier: [')
% fprintf('%.2f  ', RHSMultiplier)
% fprintf(']\n')
setting_fairness_output = strcat(sprintf('%.0f', prob_fairness(1)*100), ...
    '_', sprintf('%.0f', prob_fairness(2)*100), ...
    '_', sprintf('%.0f', prob_fairness(3)*100), ...
    '_', sprintf('%.0f', prob_fairness(4)*100), ...
    '_', sprintf('%.0f', prob_fairness(5)*100));
assert(sum(prob_fairness)==1) % Sum of probs in dist should be 1
% VOT = 28/60; % Value of Time (Business)
perc_companies = [0.5; 0.5]; % Percentage of drivers in each company
% fprintf('perc_companies: [')
% fprintf('%.2f  ', perc_companies)
% fprintf(']\n')
% setting_perc_companies = strcat(sprintf('%.0f', perc_companies(1)*100), ...
%                                 '_', sprintf('%.0f', perc_companies(2)*100));
assert(sum(perc_companies)==1) % Sum of probs in dist should be 1

%% Load Incentivization Results
setting_output_ADMM = sprintf('%.0f', nonuser_perc_ADMM(1)*100);
inputFolder0 = fullfile('data', region_, setting_region);

folderStr = 'Det_initAll2_MultT';
folderRun = fullfile(inputFolder0, ...
    strcat(folderStr, ...
    '_b', num2str(budget), ...
    '_sD', num2str(seedData), ...
    '_sA', num2str(seedADMM), ...
    '_r', num2str(rho), ...
    '_it', num2str(MaxIter2),...
    '_VOT', num2str(VOT), ...
    '_nC', num2str(n_companies), ...
    '_f', setting_fairness_output, ...
    '_percC', setting_perc_companies, ...
    '_initSB_', LogicalStr{initializeSBaseline + 1}, ...
    '_percNonU', setting_output_ADMM, ...
    '_nTIS', num2str(n_time_inc_start_ADMM), '_nTIE', num2str(n_time_inc_end_ADMM), ...
    '_ss', num2str(step_size), ...
    '_itN', num2str(iterRun)));

fileIncentivized = fullfile(folderRun, ...
    strcat('result_',...
    num2str(iterRun), ...
    '.mat'));

load(fileIncentivized, 'S_binary_full*', 'num_path_v', 'gamma', 'etaDet', 'alpha', 'n_driver_company', 'B');


%% Arrays of travel time Deterministic
% TT of each path. Size: (max_num_path*n_time)x(n_OD)
date_tt = "2018-05-01";
tt_path_incentivized = readmatrix(fullfile(folderRun, strcat('tt', date_tt, '.csv')))/60; % Seconds to minutes

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
if n_companies_cost==n_companies
    for iter_company=1:n_companies_cost
        ttDecrease(iter_company, 1) = delta_p_incentivized(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM)'*blkdiag(S_binary_full{iter_company, :})*ones(n_driver_company{iter_company}, 1)-gamma(iter_company);
        cost_v1(iter_company, 1) = alpha(iter_company).*(delta_p_incentivized(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM)'*blkdiag(S_binary_full{iter_company, :})*ones(n_driver_company{iter_company}, 1)-gamma(iter_company));
        fprintf('Cost of company %i V1: %.6f\n', iter_company, cost_v1(iter_company, 1))
        fprintf('tt change of company %i (hour): %.6f\n', iter_company, ttDecrease(iter_company, 1)/60)
    end
elseif n_companies==1
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
%         fprintf('# of drivers of company %i: %i\n', iter_company, n_c_temp)
        selected_users_temp = randsample(remained_users, n_c_temp);
        remained_users = setdiff(remained_users, selected_users_temp);
        gamma_new(iter_company) = (B(selected_users_temp, :)*etaDet)'*ones(n_c_temp, 1);
        ttDecrease(iter_company, 1) = delta_p_incentivized(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM)'*S_blkdiag(:, selected_users_temp)*ones(n_driver_company_new{iter_company}, 1)-gamma_new(iter_company);
        cost_v1(iter_company, 1) = alpha_new(iter_company).*(delta_p_incentivized(n_path*(n_time_inc_start_ADMM-1)+1:n_path*n_time_inc_end_ADMM)'*S_blkdiag(:, selected_users_temp)*ones(n_driver_company_new{iter_company}, 1)-gamma_new(iter_company));
%         if cost_v1(iter_company, 1)~=0
%             fprintf('Cost of company %i V1: %.6f\n', iter_company, cost_v1(iter_company, 1))
%         end
%         if ttDecrease(iter_company, 1)~=0
%             fprintf('Travel time change of company %i (hour): %.6f\n', iter_company, ttDecrease(iter_company, 1)/60)
%         end
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
    
