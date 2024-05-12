function ADMM_ILP(iterRun, nonuser_perc_prob, budget, n_iter_ADMM, ...
    n_time, n_time_inc_start_ADMM, n_time_inc_end_ADMM, ...
    VOT, seedADMM, seedData, rho,...
    fairness, step_size, region_, setting_region, fileName0, folderName0)
% # of companies of ADMM based algorithm. It should stay 1. 
% Assignment of drivers to companies happens randomly after solving the problem.
n_companies_ADMM = 1; 
% n_companies_UEData = 2; % # of companies
fprintf('n_companies_ADMM: %i\n', n_companies_ADMM)
% region_ = 'region_toy';
% region_ = 'region_y3';
% region_ = 'region_y3_div';
% setting_region = '5_22_AVG5_th1_pad_MultipleTimes';
% setting_region = '6_9_AVG5_th1_pad_MultipleTimes';
% n_time = 12*3;
% n_time_inc_start_ADMM = 13;
% n_time_inc_end_ADMM = 48;
fprintf('n_iter_ADMM: %i\n', n_iter_ADMM)
% VOT = 28/60;
% VOT = 2.63;
fprintf('VOT: %.6f\n', VOT)
iterRun = iterRun+1;
% nonuser_perc_prob = 0.8;
nonuser_prob = nonuser_perc_prob/100;
% prob_fairness = [0/100, 0/100, 0/100, 100/100, 0/100];
% Split the string at each underscore
strArray = split(fairness, '_');
% Convert the split string array to double precision numbers
prob_fairness = transpose(str2double(strArray));
% seedADMM = 2;
% seedData = 2;
% rho = 20;
fprintf('rho: %i\n', rho)
LogicalStr = {'F', 'T'};
perc_companies = [0.5; 0.5]; % Percentage of drivers in each company
fprintf('perc_companies: [')
fprintf('%.2f  ', perc_companies)
fprintf(']\n')
% setting_perc_companies = strcat(sprintf('%.0f', perc_companies(1)*100), ...
%     '_', sprintf('%.0f', perc_companies(2)*100));
initializeSBaseline = true;
fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})

startTime0 = tic; % Start time of running time of whole code
% LogicalStr = {'F', 'T'};
%% Initialization 1
RHSMultiplier = [1, 1.1, 1.5, 2, 2.5];
% n_iter_ADMM = n_iter_ADMM + seedADMM; % # of iterations of ADMM
permutation = true;
% Percentace of nonuser drivers in each time interval
nonuser_perc_ADMM = repmat(nonuser_prob, n_time, 1);
% Run LP approximation within the last LPWindow iterations with step of LPStep
LPWindow = 1;
LPStep = 1;
fprintf('permutation: %s\n', LogicalStr{permutation + 1})
% budget = 100 % budget of incentivization
fprintf('LPWindow: %i\n', LPWindow)
fprintf('LPStep: %i\n', LPStep)
% Print norms every normWindow steps
printNorm = false;
fprintf('printNorm: %s\n', LogicalStr{printNorm + 1})
normWindow = 100;
fprintf('normWindow: %i\n', normWindow)
% Save LP approximations of last iterations
saveLastIters = false;
fprintf('saveLastIters: %s\n', LogicalStr{saveLastIters + 1})
% Initialize the decision matrix S with the baseline
initializeSBaseline = true;
fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})
% seedADMM = 2
% rng(seedADMM) % Specification of seed for the random number generator (rng)
% Distribution of driver's tt upperbound
% prob_fairness = [0/100, 0/100, 100/100, 0/100, 0/100] % [x1.0, x1.1, x1.5, x2.0, x2.5]
fprintf('RHSMultiplier: [')
fprintf('%.2f  ', RHSMultiplier)
fprintf(']\n')
% setting_fairness_output = strcat(sprintf('%.0f', prob_fairness(1)), ...
%                             '_', sprintf('%.0f', prob_fairness(2)), ...
%                             '_', sprintf('%.0f', prob_fairness(3)), ...
%                             '_', sprintf('%.0f', prob_fairness(4)), ...
%                             '_', sprintf('%.0f', prob_fairness(5)));
assert(sum(prob_fairness./100)==1) % Sum of probs in dist should be 1
alpha = ones(n_companies_ADMM, 1)*VOT; % Value of time
perc_companies = [0.5; 0.5]; % Percentage of drivers in each company
fprintf('perc_companies: [')
fprintf('%.2f  ', perc_companies)
fprintf(']\n')
% setting_perc_companies = strcat(sprintf('%.0f', perc_companies(1)*100), ...
%                                 '_', sprintf('%.0f', perc_companies(2)*100));
assert(sum(perc_companies)==1) % Sum of probs in dist should be 1
no_obj = false; % Not having objective function (total travel time)
% no_obj = true;
if no_obj
    no_obj_str = 'n';
else
    no_obj_str = 'w';
end
fprintf('no_obj: %s\n', LogicalStr{no_obj + 1})
maxIterBisection = 50; % # of iterations of bisection
fprintf('maxIterBisection: %i\n', maxIterBisection)
error_bisection = 1e-4; % Bisection error
fprintf('error_bisection: %f\n', error_bisection)
min_bis = 0; % Min of domain in bisection
fprintf('min_bis: %i\n', min_bis)
max_bis = 600; % Max of domain in bisection
fprintf('max_bis: %i\n', max_bis)
lambda_Z_list = [0.1, 1, 5, 9]; % Multiplier of regularizer
fprintf('lambda_Z_list: [')
fprintf('%.2f  ', lambda_Z_list)
fprintf(']\n')
% Insert S of baseline for S in ADMM
insertS = false;
fprintf('insertS: %s\n', LogicalStr{insertS + 1})
insertionStep = 2000;
fprintf('insertionStep: %i\n', insertionStep)
lastInsertion = 10000;
fprintf('lastInsertion: %i\n', lastInsertion)
%% Load preprocessed data
% fileName = '_StartHour_7_AVG15_pad_theta1e+00';
setting_output_ADMM = sprintf('%.0f', nonuser_perc_ADMM(1)*100);
inputFolder0 = fullfile('../data', region_, setting_region);
folderRun0 = fullfile(inputFolder0, ...
    strcat('Det_MultT', ...
            '_sD', num2str(seedData), ...
            '_ss', num2str(step_size)));
outputFolder2 = fullfile(folderRun0, strcat('Run_', num2str(iterRun), '_initEta'));
fileName2 = fullfile(outputFolder2, strcat('data_', num2str(iterRun), '.mat'));
load(fileName2, 'n_OD', 'n_link', 'I_minTT', 'R', 'R_matrix', 'D');
% load(fileName2, 'n_OD', 'n_link', 'I_minTT', 'D');
% load(fileName2);

% file_run_2_years_ago = fullfile(inputFolder0, 'result_251_2_years_ago_run.mat');
% load(file_run_2_years_ago, 'R', 'R_matrix', 'n_OD', 'n_link', 'I_minTT', 'D');

link_loc = readmatrix(fullfile('../data', region_, 'link_loc.txt'));
% Convert start idx from 0 to 1
link_loc(:, 1) = link_loc(:, 1) + 1;
link_loc(:, 2) = link_loc(:, 2) + 1;

%% Create Save Folder
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
    '_nTIS', num2str(n_time_inc_start_ADMM), '_nTIE', num2str(n_time_inc_end_ADMM), ...
    '_ss', num2str(step_size), ...
    '_itN', num2str(iterRun)));
if n_iter_ADMM >= 0    
    mkdir(folderRun);
    log_file = fullfile(folderRun, 'outputLog.txt');
    diary(log_file); % Start logging to a file named 'outputLog.txt'
end
fprintf('n_iter_ADMM: %i\n', n_iter_ADMM)
fprintf('VOT: %.6f\n', VOT)
fprintf('n_companies_ADMM: %i\n', n_companies_ADMM)
fprintf('rho: %i\n', rho)
fprintf('perc_companies: [')
fprintf('%.2f  ', perc_companies)
fprintf(']\n')
fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})
fprintf('permutation: %s\n', LogicalStr{permutation + 1})
fprintf('LPWindow: %i\n', LPWindow)
fprintf('LPStep: %i\n', LPStep)
fprintf('printNorm: %s\n', LogicalStr{printNorm + 1})
fprintf('normWindow: %i\n', normWindow)
fprintf('saveLastIters: %s\n', LogicalStr{saveLastIters + 1})
fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})
fprintf('RHSMultiplier: [')
fprintf('%.2f  ', RHSMultiplier)
fprintf(']\n')
fprintf('perc_companies: [')
fprintf('%.2f  ', perc_companies)
fprintf(']\n')
fprintf('no_obj: %s\n', LogicalStr{no_obj + 1})
fprintf('maxIterBisection: %i\n', maxIterBisection)
fprintf('error_bisection: %f\n', error_bisection)
fprintf('min_bis: %i\n', min_bis)
fprintf('max_bis: %i\n', max_bis)
fprintf('lambda_Z_list: [')
fprintf('%.2f  ', lambda_Z_list)
fprintf(']\n')
fprintf('insertS: %s\n', LogicalStr{insertS + 1})
fprintf('insertionStep: %i\n', insertionStep)
fprintf('lastInsertion: %i\n', lastInsertion)
%% Load A for reducing volume of users
inputFolder1 = fullfile('../data', region_, setting_region, 'initialData');
folderRunDet = fullfile('../data', region_, setting_region, ...
                    strcat('Det_MultT', ...
                    '_sD', num2str(seedData), ...
                    '_ss', num2str(step_size)));
if ~(iterRun==1)
    inputFolder2Det = fullfile(folderRunDet, strcat('Run_', num2str(iterRun)));
    inputFolder2Det_initEta = fullfile(folderRunDet, strcat('Run_', num2str(iterRun), '_initEta'));
end
% A = cell(n_time, 1);
% % A2check = cell(n_time, 1);
% P = cell(n_time, 1);
% for iter_time=1:n_time
%     if iterRun==1
%         A{iter_time, 1} = readmatrix(fullfile(inputFolder1, strcat('A', num2str(iter_time), fileName0, '.txt')));
%     else
%         A{iter_time, 1} = readmatrix(fullfile(inputFolder2Det, strcat('A', num2str(iter_time), fileName0, '.txt')));
%         %         P{iter_time, 1} = readmatrix(fullfile(inputFolder2Det, strcat('P', num2str(iter_time), fileName0, '.txt')));
%         %         A2check{iter_time, 1} = R{iter_time, 1}*P{iter_time, 1};
%         %         fprintf(strcat(num2str(norm(A{iter_time, 1}-A2check{iter_time, 1})), '\n'));
%     end
% end
%% Initialize vector of companies
rng(seedData)
% inputFolder1 = fullfile(inputFolder0, 'initialData');
q_time = cell(n_time, 1);
q_nonuser_time = cell(n_time, 1);
q_user_time = cell(n_time, 1);
q_nonuser_time(:) = {[]};
q_user_time(:) = {[]};
m = cell(n_time, 1); % Keeps track of number of ODS (or roads?)
n_user = cell(n_time, 1);
n_nonuser = cell(n_time, 1);
for iter_time=1:n_time
    q_time{iter_time, 1} = readmatrix(fullfile(inputFolder1, strcat('q', num2str(iter_time), fileName0, '.txt')));
    n_t = sum(q_time{iter_time, 1});
    n_nonuser_temp100 = n_t;
    selected_nonusers100 = randsample(1:n_t,n_nonuser_temp100);
    n_nonuser_temp = round(nonuser_prob*n_t);
%     selected_nonusers = randsample(1:n_t,n_nonuser_temp);
    selected_nonusers = selected_nonusers100(1:n_nonuser_temp);
%     n_nonuser_temp90 = round(0.9*n_t);
%     selected_nonusers90 = randsample(1:n_t,n_nonuser_temp90);
%     n_nonuser_temp = round(nonuser_prob*n_t);
% %     selected_nonusers = randsample(1:n_t,n_nonuser_temp);
%     selected_nonusers = selected_nonusers90(1:n_nonuser_temp);
    counter_driver = 0;
    if iter_time>=n_time_inc_start_ADMM && iter_time<=n_time_inc_end_ADMM
        for iter_OD_time=1:n_OD
            n_nonuser_OD_time = 0;
            for idx_driver = 1:q_time{iter_time, 1}(iter_OD_time)
                counter_driver = counter_driver + 1;
                n_nonuser_OD_time = n_nonuser_OD_time + sum(ismember(selected_nonusers, counter_driver));
            end
            q_nonuser_time{iter_time, 1} = [q_nonuser_time{iter_time, 1}; n_nonuser_OD_time];
            n_user_OD_time = q_time{iter_time, 1}(iter_OD_time) - n_nonuser_OD_time;
            q_user_time{iter_time, 1} = [q_user_time{iter_time, 1}; n_user_OD_time];
            assert(q_time{iter_time, 1}(iter_OD_time)==n_nonuser_OD_time+n_user_OD_time);
        end
    else
        for iter_OD_time=1:n_OD
            if q_time{iter_time, 1}(iter_OD_time)>0
                q_nonuser_time{iter_time, 1} = [q_nonuser_time{iter_time, 1}; q_time{iter_time, 1}(iter_OD_time)];
            else
                q_nonuser_time{iter_time, 1} = [q_nonuser_time{iter_time, 1}; 0];
            end
            q_user_time{iter_time, 1} = [q_user_time{iter_time, 1}; 0];
        end
    end
    m{iter_time, 1} = size(R{iter_time, 1}, 2);
    n_user{iter_time, 1} = sum(q_user_time{iter_time, 1});
    n_nonuser{iter_time, 1} = sum(q_nonuser_time{iter_time, 1});
end
q = cat(1, q_time{:});
q_user = cat(1, q_user_time{:});
q_nonuser = cat(1, q_nonuser_time{:});
% q vectors of each company. It includes the # of drivers between ODs
% size q_company{., .} = n_OD*n_time
q_company = cell(n_companies_ADMM, 1);
q_company(:) = {[]};
n_t = sum(q_user);
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% Create q vector for each company. It contains the number of company
% drivers at each time between each OD
n_c_floor = floor(1/n_companies_ADMM*n_t);
n_c_round = n_c_floor*n_companies_ADMM;
n_c_mod = mod(n_t, n_c_round);
remained_users = 1:n_t;
for iter_company=1:n_companies_ADMM
    if iter_company<=n_c_mod
        n_c_temp = n_c_floor + 1;
    else
        n_c_temp = n_c_floor;
    end
    selected_users_temp = randsample(remained_users, n_c_temp);
    remained_users = setdiff(remained_users, selected_users_temp);
    counter_driver1 = 1;
    counter_driver2 = 0;
    for iter_OD_time=1:size(q_user, 1)
        if q_user(iter_OD_time)>0
            counter_driver2 = counter_driver2 + q_user(iter_OD_time);
            n_c_OD_time = sum(ismember(selected_users_temp, counter_driver1:counter_driver2));
            counter_driver1 = counter_driver2 + 1;
            q_company{iter_company} = [q_company{iter_company}; n_c_OD_time];
        else
            q_company{iter_company} = [q_company{iter_company}; 0];
        end
    end
end
% Is (# if drivers) == (# of users)?
q_sum_check=0;
for iter_company=1:n_companies_ADMM
    q_sum_check = q_sum_check + q_company{iter_company};
end
assert(sum(abs(q_user-q_sum_check))==0);
clearvars n_user_temp_check q_sum_check
% q vectors of each company at each time
% size q_company_time{., .} = n_OD
q_company_time = cell(n_companies_ADMM, n_time);
for iter_company=1:n_companies_ADMM
    q_temp = q_company{iter_company};
    for iter_time=1:n_time
        q_company_time{iter_company, iter_time} = q_temp(1+(iter_time-1)*n_OD:iter_time*n_OD, 1);
    end
end
clearvars q_temp
% # of drivers of each company
% size n_driver_company{., .} = 1
n_driver_company = cell(n_companies_ADMM, 1);
for iter_company=1:n_companies_ADMM
    n_driver_company{iter_company} = sum(q_company{iter_company});
end
% 
%% Arrays of travel time Deterministic
% # of paths of each OD
num_path_v = readmatrix(fullfile(inputFolder1, 'num_path_v.txt'));
% TT of each path. Size: (max_num_path*n_time)x(n_OD)
if iterRun==1
%     tt_pathDet = readmatrix(fullfile(inputFolder1, strcat('tt', fileName0, '.txt')))/60; % Seconds to minutes
    tt_pathDet_initEta = readmatrix(fullfile(inputFolder1, strcat('tt', fileName0, '.txt')))/60; % Seconds to minutes
else
%     tt_pathDet = readmatrix(fullfile(inputFolder2Det, strcat('tt', fileName0, '.txt')))/60; % Seconds to minutes
    tt_pathDet_initEta = readmatrix(fullfile(inputFolder2Det_initEta, strcat('tt', fileName0, '.txt')))/60; % Seconds to minutes
end
max_num_path = max(num_path_v);
n_path = sum(num_path_v);
n_OD = size(tt_pathDet_initEta, 2);
etaDet = zeros(n_OD*n_time, 1); 
delta_pDet = zeros(n_path*n_time, 1);
I_minTT_initEta = cell(n_time, 1); 
I_minTT_initEta(:) = {zeros(n_OD, 1)};
for iter_time=1:n_time
    idxTemp = 0; % Starting idx of path
    % Create delta_pDet
    index_0 = tt_pathDet_initEta(max_num_path*(iter_time-1)+1, :)==0;
    tt_pathDet_initEta(max_num_path*(iter_time-1)+1, index_0) = 0.0001;
    temp_delta = tt_pathDet_initEta(max_num_path*(iter_time-1)+1:max_num_path*(iter_time), :);
    temp_delta = temp_delta(:);
    temp_delta(temp_delta==0) = [];
    delta_pDet(n_path*(iter_time-1)+1:n_path*(iter_time)) = temp_delta;
    
    % Create etaDet
    row_index = max_num_path*(iter_time-1)+1:max_num_path*(iter_time);
    for iter_OD=1:n_OD
        tempMinTT = tt_pathDet_initEta(row_index, iter_OD);
        tempMinTT(tempMinTT==0) = [];
        [tempMinTT, tempMinTTIdx] = min(tempMinTT);
        etaDet((iter_time-1)*n_OD+iter_OD, 1) = tempMinTT;
        I_minTT_initEta{iter_time, 1}(iter_OD, 1) = idxTemp + tempMinTTIdx;
        idxTemp = idxTemp + num_path_v(iter_OD, 1);
    end
end
delta_pDet(delta_pDet==0.0001) = 0;
etaDet(etaDet==0.0001) = 0;
clearvars index_0 temp_delta row_index tempMinTT iter_time iter_OD
%% User decision matrix
S_user = cell(n_time, 1);
S_nonuser = cell(n_time, 1);
v_userDet = cell(n_time, 1);
v_userDet(:) = {zeros(n_link*n_time, 1)};
% v_userStochastic = cell(n_time, 1);
% v_userStochastic(:) = {zeros(n_link*n_time, 1)};
v_nonuser = cell(n_time, 1);
v_nonuser(:) = {zeros(n_link*n_time, 1)};
% v_dynamic_array = zeros(n_link*n_time, 1); % users' volume of links
% v_dynamic_array_NoInc_Det = zeros(n_link*n_time, 1); % users' volume of links  where company drivers act deterministic
v_dynamic_array_NoInc_allDet = zeros(n_link*n_time, 1); % users' volume of links where all drivers act stochastic
v_tilda_baseline = zeros(n_link*n_time, 1); % nonusers' volume of links
for iter_time=1:n_time
    S_user{iter_time} = zeros(m{iter_time, 1}, n_user{iter_time, 1});
    S_nonuser{iter_time} = zeros(m{iter_time, 1}, n_nonuser{iter_time, 1});
    counter_driver_user = 1;
    counter_driver_nonuser = 1;
    for iter_OD=1:n_OD
        %         index_min = I(iter_OD);
        index_min = I_minTT{iter_time, 1}(iter_OD, 1); % !!!!!!!! iter_time=1 here <<>> Old setting
        for iter_driver=1:q_user_time{iter_time, 1}(iter_OD, 1)
            S_user{iter_time}(index_min, counter_driver_user) = 1;
            counter_driver_user = counter_driver_user + 1;
        end
        for iter_driver=1:q_nonuser_time{iter_time, 1}(iter_OD, 1)
            S_nonuser{iter_time}(index_min, counter_driver_nonuser) = 1;
            counter_driver_nonuser = counter_driver_nonuser + 1;
        end
    end
    v_userDet{iter_time, 1} = zeros(n_link*n_time, 1);
    v_userDet{iter_time, 1}(n_link*(iter_time-1)+1:end, 1) = R{iter_time, 1} * ...
        S_user{iter_time} * ones(size(S_user{iter_time}, 2), 1);
    %     v_userStochastic{iter_time, 1} = zeros(n_link*n_time, 1);
    %     v_userStochastic{iter_time, 1}(n_link*(iter_time-1)+1:end, 1) = A{iter_time, 1} * ...
    %         S_user{iter_time} * ones(size(S_user{iter_time}, 2), 1);
    v_nonuser{iter_time, 1} = zeros(n_link*n_time, 1);
    v_nonuser{iter_time, 1}(n_link*(iter_time-1)+1:end, 1) = R{iter_time, 1} * ...
        S_nonuser{iter_time} * ones(size(S_nonuser{iter_time}, 2), 1);
    %     v_dynamic_array = v_dynamic_array + v_userDet{iter_time, 1};
    %     v_dynamic_array_NoInc_Det = v_dynamic_array_NoInc_Det + v_userDet{iter_time, 1};
    v_dynamic_array_NoInc_allDet = v_dynamic_array_NoInc_allDet + v_userDet{iter_time, 1};
    v_tilda_baseline = v_tilda_baseline + v_nonuser{iter_time, 1};
end


%% Initialization 2
% # of drivers of each company at each time
% size n_driver_company_time{., .} = 1
for iter_time=1:n_time
    n_driver_time = sum(q_time{iter_time});
    n_driver_user_time = sum(q_user_time{iter_time});
    n_driver_nonuser_time = sum(q_nonuser_time{iter_time});
    fprintf('# of drivers @ time %i: total=%i, user=%i (%g%%), nonuser=%i (%g%%)\n', ...
        iter_time, n_driver_time, ... 
        n_driver_user_time, n_driver_user_time/n_driver_time*100, ...
        n_driver_nonuser_time, n_driver_nonuser_time/n_driver_time*100)
%     fprintf('Percentage of drivers @ time %i: user=%g, nonuser=%g\n', ...
%         iter_time, n_driver_user_time/n_driver_time*100, n_driver_nonuser_time/n_driver_time*100)
end
n_driver_company_time = cell(n_companies_ADMM, n_time);
for iter_company=1:n_companies_ADMM
    for iter_time=1:n_time
        n_driver_company_time{iter_company, iter_time} = sum(q_company_time{iter_company, iter_time});
        fprintf('# of drivers of company %i @ time %i: %i\n', ...
            iter_company, iter_time, n_driver_company_time{iter_company, iter_time})
    end
end
% b
% size b{iter_company} = n_driver_company{iter_company}
b = cell(n_companies_ADMM, 1);
for iter_company=1:n_companies_ADMM
    if sum(prob_fairness) == 1
        b{iter_company} = RHSMultiplier(discretize(rand(1,n_driver_company{iter_company}),cumsum([0,prob_fairness])))';
    elseif sum(prob_fairness) == 100
        b{iter_company} = RHSMultiplier(discretize(rand(1,n_driver_company{iter_company}),cumsum([0,prob_fairness./100])))';
    end
end

% % b at each time
% % size b_time{iter_company, iter_time} = n_driver_company_time{iter_company, iter_time}
% b_time = cell(n_companies_ADMM, n_time);
% for iter_company=1:n_companies_ADMM
%     b_temp = b{iter_company};
%     idxStart = 1;
%     idxEnd = n_driver_company_time{iter_company, 1};
%     for iter_time=1:n_time
%         b_time{iter_company, iter_time} = b_temp(idxStart:idxEnd, 1);
%         if iter_time~=n_time
%             idxStart = idxEnd + 1;
%             idxEnd = idxEnd + n_driver_company_time{iter_company, iter_time+1};
%         end
%     end
% end

% B
% size B{iter_company} = (n_driver_company{iter_company}, n_OD*n_time)
B = cell(n_companies_ADMM, 1);
for iter_company=1:n_companies_ADMM
    B_temp = zeros(n_driver_company{iter_company}, n_OD*n_time);
    index_row_start = 1;
    for iter_q=1:size(q, 1)
        if q_company{iter_company}(iter_q, 1) > 0
            index_row_end = index_row_start + q_company{iter_company}(iter_q, 1) - 1;
            B_temp(index_row_start:index_row_end, iter_q) = 1;
            index_row_start = index_row_end+1;
        end
    end
    B{iter_company} = B_temp;
end

% % B at each time
% B_time = cell(n_companies_ADMM, n_time);
% for iter_company=1:n_companies_ADMM
%     for iter_time=1:n_time
%         B_temp = zeros(n_driver_company_time{iter_company, iter_time}, n_OD);
%         index_row_start = 1;
%         for iter_q=1:size(q, 1)/n_time
%             if q_company_time{iter_company, iter_time}(iter_q, 1) > 0
%                 index_row_end = (index_row_start - 1) + q_company_time{iter_company, iter_time}(iter_q, 1);
%                 B_temp(index_row_start:index_row_end, iter_q) = 1;
%                 index_row_start = index_row_end + 1;
%             end
%         end
%         B_time{iter_company, iter_time} = B_temp;
%     end
% end

% gamma
gamma = [];
for iter_company=1:n_companies_ADMM
    gamma = [gamma; (B{iter_company}*etaDet)' * ones(size(B{iter_company}, 1), 1)];
end
% RHS of fairness constraint
RHS = cell(n_companies_ADMM, 1);
for iter_company=1:n_companies_ADMM
    RHS{iter_company} = b{iter_company}.*(B{iter_company} * etaDet);
end


% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% Indecies of S blocks
idxEndCell = cell(n_companies_ADMM, n_time);
for iter_companies=1:n_companies_ADMM
    idxEndCell{iter_companies, 1} = n_driver_company_time{iter_companies, 1};
    for iter_time=2:n_time
        idxEndCell{iter_companies, iter_time} = idxEndCell{iter_companies, iter_time-1} + ...
            n_driver_company_time{iter_companies, iter_time};
    end
end

ones_row = ones(n_path*n_time, 1);
ones_S = cell(n_companies_ADMM, n_time);
for iter_company=1:n_companies_ADMM
    for iter_time=1:n_time
        ones_S{iter_company, iter_time} = ones(n_driver_company_time{iter_company, iter_time}, 1);
    end
end
%% Load BPR constants
% Free flow travel time
tt0_array = readmatrix(fullfile('../data', 'capacity', region_, strcat(folderName0, 'link_tt_0_minutes_', region_, '.csv')));
% Capacity of link
w_array = readmatrix(fullfile('../data', 'capacity', region_, strcat(folderName0, 'link_capacity_', region_, '.csv')));

%% Functions
% Travel time function of the link given the volume (based on BPR funciton)
F_omega = @(omegaTemp, tt0, wTemp) tt0*omegaTemp + (0.15*tt0/(wTemp^4))*(omegaTemp^5);
% L_omega = @(omegaTemp, v, tt0, wTemp, l, r, rhoTemp, uTemp) tt0*omegaTemp + (0.15*tt0/(wTemp^4))*(omegaTemp^5) + l*(omegaTemp-r*uTemp-v) + rhoTemp/2*((omegaTemp-r*uTemp-v)^2);
% Gradient of objective function wrt omega
gradient_omega = @(omegaTemp, v, tt0, wTemp, l, r, rhoTemp, uTemp) tt0 + 5*(0.15*tt0/(wTemp^4))*(omegaTemp^4) + l + rhoTemp*(omegaTemp-r*uTemp-v);


%% Initialize variables
rng(seedADMM)
D = sparse(D);
R_matrix = sparse(R_matrix);
omega = zeros(size(R_matrix, 1), 1);
c_tilda = zeros(n_companies_ADMM*2, 1);
beta_tilda = 0;
eye_tilda = [eye(n_companies_ADMM), -eye(n_companies_ADMM)];
one_tilda = [ones(n_companies_ADMM, 1); zeros(n_companies_ADMM, 1)];
Tempc_tilda = (1/rho)*pinv(eye_tilda'*eye_tilda + one_tilda*one_tilda');
delta_p_t = cell(n_time, 1);
for iter_time=1:n_time
    delta_p_t{iter_time, 1} = delta_pDet((iter_time-1)*n_path+1:iter_time*n_path, 1);
end
if n_iter_ADMM < 0
    if n_iter_ADMM == -1
        solver_name = "Gurobi"
    elseif n_iter_ADMM == -2
        solver_name = "Mosek"
    elseif n_iter_ADMM == -3
        solver_name = "GLPK"
    end
    folderRunSolver = fullfile(inputFolder0, ...
                        strcat(solver_name, '_new_Det_initAll2_MultT', ...
                                '_b', num2str(budget), ...
                                '_sD', num2str(seedData), ...
                                '_sS', num2str(seedADMM), ...
                                '_VOT', num2str(VOT), ...
                                '_nC', num2str(1), ...
                                '_f', fairness, ...
                                '_percNonU', num2str(setting_output_ADMM), ...
                                '_nTIS', num2str(n_time_inc_start_ADMM), ...
                                '_nTIE', num2str(n_time_inc_end_ADMM), ...
                                '_ss', num2str(step_size), ...
                                '_itN', num2str(iterRun)));
    fprintf('folderRunSolver: %s\n', folderRunSolver)
    
    fileSolverData = fullfile(folderRunSolver,...
        strcat('result_', num2str(iterRun), '_ADMMDataInit.mat'));
    save(fileSolverData);
    
    % Beep sound
    sound(sin(1:3000));
    return;
end
delta_Cell = repmat({delta_pDet}, 1, n_companies_ADMM);
Delta = blkdiag(delta_Cell{:});
D_tilda_Cell = repmat({D}, 1, n_companies_ADMM);
D_tilda = sparse(blkdiag(D_tilda_Cell{:}));
R_tilda = sparse(repmat(R_matrix, 1, n_companies_ADMM));
D_transpose_D_Cell = repmat({D'*D}, 1, n_companies_ADMM);
D_tilda_transpose_D_tilda = blkdiag(D_transpose_D_Cell{:});
Delta_alpha = Delta*diag(alpha);
Tempu = (1/rho)*(eye(n_path*n_time*n_companies_ADMM) + repmat(R_matrix'*R_matrix, n_companies_ADMM, n_companies_ADMM) + ...
    D_tilda_transpose_D_tilda + Delta_alpha*Delta_alpha')^(-1);
clearvars D_transpose_D_Cell D_tilda_transpose_D_tilda
Tempu = sparse(Tempu);
TempW = (1/rho)*(ones(n_path*n_time) + eye(n_path*n_time))^(-1);
TempH = (1/rho)*(delta_pDet*delta_pDet' + eye(size(delta_pDet, 1)))^(-1);
TempW_time = (1/rho)*(ones(n_path) + eye(n_path))^(-1);
TempH_time = cell(n_time, 1);
for iter_time=1:n_time
    TempH_time{iter_time, 1} = (1/rho)*(delta_p_t{iter_time, 1}*delta_p_t{iter_time, 1}' + eye(size(delta_p_t{iter_time, 1}, 1)))^(-1);
end
% Company i
S = cell(n_companies_ADMM, 1);
S_time = cell(n_companies_ADMM, n_time);
W = cell(n_companies_ADMM, 1);
W_time = cell(n_companies_ADMM, n_time);
H = cell(n_companies_ADMM, 1);
H_time = cell(n_companies_ADMM, n_time);
Z = cell(n_companies_ADMM, 1);
Z_time = cell(n_companies_ADMM, n_time);
beta = cell(n_companies_ADMM, 1);
beta_time = cell(n_companies_ADMM, n_time);
u = cell(n_companies_ADMM, 1);
u_vectorized = zeros(n_companies_ADMM*n_path*n_time, 1);
TempS = cell(n_companies_ADMM, 1);
TempS_time = cell(n_companies_ADMM, n_time);
S_user_company_time = cell(n_companies_ADMM, n_time);
for iter_company=1:n_companies_ADMM
    if initializeSBaseline
        for iter_time=1:n_time
            n_user_temp = sum(q_company_time{iter_company, iter_time});
            S_user_company_time{iter_company, iter_time} = zeros(n_path, n_user_temp);
            counter_driver_user = 1;
            for iter_OD=1:n_OD
%                 index_min = I(iter_OD);
                index_min = I_minTT{iter_time, 1}(iter_OD, 1); % !!!!!!!! iter_time=1 here <<>> Old setting
                for iter_driver=1:q_company_time{iter_company, iter_time}(iter_OD, 1)
                    S_user_company_time{iter_company, iter_time}(index_min, counter_driver_user) = 1;
                    counter_driver_user = counter_driver_user + 1;
                end
            end
        end
        S{iter_company} = blkdiag(S_user_company_time{iter_company, :});
    else
        S{iter_company} = zeros(n_path*n_time, n_driver_company{iter_company});
    end
    W{iter_company} = S{iter_company};
    H{iter_company} = S{iter_company};
    Z{iter_company} = S{iter_company};
    beta{iter_company} = zeros(n_driver_company{iter_company}, 1);
    u{iter_company} = zeros(n_path*n_time, 1);
    TempS{iter_company} = (1/rho)*(ones(size(S{iter_company}, 2))+3*eye(size(S{iter_company}, 2)))^(-1);
    for iter_time=1:n_time
        S_time{iter_company, iter_time} = zeros(n_path, n_driver_company_time{iter_company, iter_time});
        W_time{iter_company, iter_time} = S_time{iter_company, iter_time};
        H_time{iter_company, iter_time} = S_time{iter_company, iter_time};
        Z_time{iter_company, iter_time} = S_time{iter_company, iter_time};
        beta_time{iter_company, iter_time} = zeros(n_driver_company_time{iter_company, iter_time}, 1);
        TempS_time{iter_company, iter_time} = (1/rho)*(ones(size(S_time{iter_company, iter_time}, 2))+3*eye(size(S_time{iter_company, iter_time}, 2)))^(-1);
    end
end

lambda1 = cell(n_companies_ADMM, 1);
lambda4 = cell(n_companies_ADMM, 1);
lambda5 = cell(n_companies_ADMM, 1);
lambda6 = cell(n_companies_ADMM, 1);
lambda8 = cell(n_companies_ADMM, 1);
lambda10 = cell(n_companies_ADMM, 1);
for iter_company=1:n_companies_ADMM
    lambda1{iter_company} = zeros(n_path*n_time, 1);
    lambda3{iter_company} = zeros(size(q_company{iter_company}));
    lambda4{iter_company} = zeros(size(W{iter_company}, 2), 1);
    lambda5{iter_company} = zeros(size(S{iter_company}));
    lambda6{iter_company} = zeros(size(beta{iter_company}));
    lambda8{iter_company} = zeros(size(H{iter_company}));
    lambda10{iter_company} = zeros(size(Z{iter_company}));
end
lambda2 = omega;
lambda7 = zeros(n_companies_ADMM, 1);
lambda9 = 0;

ConstViolation = zeros(n_iter_ADMM, 1);
ConstViolation_normalized = zeros(n_iter_ADMM, 1);
Augmented_Lagrangian = zeros(n_iter_ADMM, 1);
total_travel_time_array = zeros(n_iter_ADMM, 1);
time_array = zeros(n_iter_ADMM, 9);
time_array_dual = zeros(n_iter_ADMM, 10);
counter_save = 1;
%% ADMM
% cvx_solver Gurobi
% cvx_solver MOSEK
% cvx_solver SDPT3
% cvx_solver SeDuMi
for iter=1:n_iter_ADMM
    %     iter
    if permutation == true
        permutation_order = randi([0, 1]);
    else
        permutation_order = 1;
    end
    if permutation_order==1
        % Block 1
        uTemp = zeros(n_path*n_time, 1);
        for iter_company=1:n_companies_ADMM
            uTemp = uTemp + u{iter_company};
        end
        F_omega_total = 0;
        if no_obj
            uStartTime = tic;
            omega = -(1/rho)*lambda2 + R_matrix*uTemp + v_tilda_baseline;
            time_array(iter, 1) = toc(uStartTime);
            for iter_omega = 1:size(omega, 1)
                tt0Temp = tt0_array(link_loc(mod(iter_omega-1, size(tt0_array, 1)) + 1, 1), 3);
                wTemp = w_array(link_loc(mod(iter_omega-1, size(w_array, 1)) + 1, 1), 3);
                F_omega_total = F_omega_total + F_omega(omega(iter_omega, 1), tt0Temp, wTemp);
            end
        else
            for iter_omega = 1:size(omega, 1)
                tt0Temp = tt0_array(link_loc(mod(iter_omega-1, size(tt0_array, 1)) + 1, 1), 3);
                wTemp = w_array(link_loc(mod(iter_omega-1, size(w_array, 1)) + 1, 1), 3);
                r = R_matrix(iter_omega, :);
                l = lambda2(iter_omega);
                vTemp = v_tilda_baseline(iter_omega);
                %  FYI:  gradient_omega = @(omega, v, tt0, w, l, r, rho, u) tt0 + 5*(0.15*tt0/(w^4))*(omega^4) + l + rho*(omega-r*u-v);
                %  FYI:  bisectionMethodNew(gradient_omega, v, min_bis, max_bis, error_bisection, tt0, w, l, a, r, u)
                uStartTime = tic;
                omega(iter_omega) = bisectionMethodNew(gradient_omega, vTemp, min_bis, max_bis, error_bisection, tt0Temp, wTemp, l, r, rho, uTemp, maxIterBisection);
                time_array(iter, 1) = time_array(iter, 1) + toc(uStartTime);
                % Our omega has constant volume in itself so v=0
                F_omega_total = F_omega_total + F_omega(omega(iter_omega, 1), tt0Temp, wTemp);
            end
        end
        total_travel_time_array(iter, 1) = F_omega_total/60;
        
        SStartTime = tic;
        if insertS && mod(iter, insertionStep)==0 && iter<=lastInsertion
            fprintf('Iteration %i >> S baseline injected', iter)
            for iter_company=1:n_companies_ADMM
                S{iter_company} = blkdiag(S_user_company_time{iter_company, :});
            end
        else
            for iter_company=1:n_companies_ADMM
                idxStart = 1;
                idxEnd = n_driver_company_time{iter_company, 1};
                for iter_time=1:n_time
                    S{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) =...
                        (-lambda1{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, 1)*ones(1,n_driver_company_time{iter_company, iter_time}) - ...
                        lambda5{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) - ...
                        lambda8{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) - ...
                        lambda10{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) + ....
                        rho*u{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, 1)*ones(1, n_driver_company_time{iter_company, iter_time}) + ...
                        rho*W{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) + ...
                        rho*H{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) + ...
                        rho*Z{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd))*TempS_time{iter_company, iter_time};
                    if iter_time~=n_time
                        idxStart = idxEnd + 1;
                        idxEnd = idxEnd + n_driver_company_time{iter_company, iter_time+1};
                    end
                end
            end
        end
        time_array(iter, 2) = toc(SStartTime);
        betaStartTime = tic;
        for iter_company=1:n_companies_ADMM
            tempRHS = b{iter_company}.*(B{iter_company}*etaDet);
            idxStart = 1;
            idxEnd = n_driver_company_time{iter_company, 1};
            for iter_time=1:n_time
                beta{iter_company}(idxStart:idxEnd, 1) = ...
                    (1/rho) * (-lambda6{iter_company}(idxStart:idxEnd, 1) - ...
                    rho*H{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd)'*delta_p_t{iter_time, 1} + ...
                    rho*tempRHS(idxStart:idxEnd, 1));
                if iter_time~=n_time
                    idxStart = idxEnd + 1;
                    idxEnd = idxEnd + n_driver_company_time{iter_company, iter_time+1};
                end
            end
            beta{iter_company}(beta{iter_company}<0) = 0;
        end
        time_array(iter, 3) = toc(betaStartTime);
        
        c_tildaStartTime = tic;
        c_tilda = Tempc_tilda * (eye_tilda'*lambda7 - lambda9*one_tilda - ...
            rho*((eye_tilda')*(alpha.*gamma)) + ...
            rho*((eye_tilda')*(alpha.*(Delta'*u_vectorized))) - ...
            rho*beta_tilda*one_tilda + rho*budget*one_tilda);
        c_tilda(c_tilda<0) = 0;
        time_array(iter, 4) = toc(c_tildaStartTime);
        
        % Block 2
        lambda1_vectorized = [];
        lambda3_vectorized = [];
        SOne_vectorized = [];
        q_company_vectorized = [];
        for iter_company=1:n_companies_ADMM
            lambda1_vectorized = [lambda1_vectorized; lambda1{iter_company}];
            lambda3_vectorized = [lambda3_vectorized; lambda3{iter_company}];
            SOne_vectorized = [SOne_vectorized; S{iter_company}*ones(size(S{iter_company}, 2), 1)];
            q_company_vectorized = [q_company_vectorized; q_company{iter_company}];
        end
        u_vectorizedStartTime = tic;
        u_vectorized = Tempu * (lambda1_vectorized + R_tilda'*lambda2 - ...
            D_tilda'*lambda3_vectorized - Delta_alpha*lambda7 + ...
            rho*SOne_vectorized - ...
            rho*R_tilda'*v_tilda_baseline + rho*R_tilda'*omega + ...
            rho*D_tilda'*q_company_vectorized + ...
            rho*(Delta_alpha)*(alpha.*gamma) + ...
            rho*(Delta_alpha)*eye_tilda*c_tilda);
        time_array(iter, 5) = toc(u_vectorizedStartTime);
        for iter_company=1:n_companies_ADMM
            u{iter_company} = u_vectorized(1+n_path*n_time*(iter_company-1):n_path*n_time*iter_company, 1);
        end
        
        for iter_company=1:n_companies_ADMM
            WStartTime = tic;
            tempRHS = b{iter_company}.*(B{iter_company}*etaDet);
            idxStart = 1;
            idxEnd = n_driver_company_time{iter_company, 1};
            for iter_time=1:n_time
                W{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) = ...
                    TempW_time*(-ones(n_path, 1)*lambda4{iter_company}(idxStart:idxEnd, 1)' + ...
                    lambda5{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) + ...
                    rho*ones(size(W{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd))) + ...
                    rho*S{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd));
                time_array(iter, 6) = time_array(iter, 6) + toc(WStartTime);
                
                HStartTime = tic;
                H{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) = ...
                    TempH_time{iter_time, 1}*(-delta_p_t{iter_time, 1}*lambda6{iter_company}(idxStart:idxEnd, 1)' + ...
                    lambda8{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) - ...
                    rho*delta_p_t{iter_time, 1}*beta{iter_company}(idxStart:idxEnd, 1)' + ...
                    rho*(delta_p_t{iter_time, 1}*(tempRHS(idxStart:idxEnd, 1))') + ...
                    rho*S{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd));
                time_array(iter, 7) = time_array(iter, 7) + toc(HStartTime);
                if iter_time~=n_time
                    idxStart = idxEnd + 1;
                    idxEnd = idxEnd + n_driver_company_time{iter_company, iter_time+1};
                end
            end
            if iter_company~=n_companies_ADMM
                idxStart = idxEnd + 1;
                idxEnd = idxEnd + n_driver_company_time{iter_company+1, 1};
            end
        end
        
        if iter<=n_iter_ADMM/6
            lambda_Z = lambda_Z_list(1);
        elseif  (iter>n_iter_ADMM/30) && (2*n_iter_ADMM/30)
            lambda_Z = lambda_Z_list(2);
        elseif (2*iter>n_iter_ADMM/30) && (4*n_iter_ADMM/30)
            lambda_Z = lambda_Z_list(3);
        else
            lambda_Z = lambda_Z_list(4);
        end
        ZStartTime = tic;
        for iter_company=1:n_companies_ADMM
            Z{iter_company} = (1/(rho-lambda_Z))*(lambda10{iter_company}+ ...
                rho*S{iter_company} - lambda_Z/2);
            if rho>lambda_Z
                Z{iter_company}(Z{iter_company}<0) = 0;
                Z{iter_company}(Z{iter_company}>1) = 1;
            else
                Z{iter_company}(Z{iter_company}<=0.5) = 1;
                Z{iter_company}(Z{iter_company}>0.5) = 0;
            end
            for iter_time=1:n_time
                if iter_time == 1
                    Z{iter_company}(1:n_path, sum(q_company{iter_company}(1:n_OD))+1:end) = 0;
                elseif iter_time == n_time
                    Z{iter_company}(n_path*(n_time-1)+1:end, 1:sum(q_company{iter_company}(1:n_OD*(n_time-1)))) = 0;
                else
                    Z{iter_company}(n_path*(iter_time-1)+1:n_path*iter_time, 1:sum(q_company{iter_company}(1:(iter_time-1)*n_OD))) = 0;
                    Z{iter_company}(n_path*(iter_time-1)+1:n_path*iter_time, sum(q_company{iter_company}(1:iter_time*n_OD))+1:end) = 0;
                end
            end
        end
        
        for iter_company=1:n_companies_ADMM
            idxEndRow1 = 0;
            idxStartCol = 1;
            idxEndCol = 0;
            for iter_time=1:n_time
                for iter_OD=1:n_OD
                    nPathTemp = num_path_v(iter_OD, 1);
                    nDriverTemp  = q_company_time{iter_company, iter_time}(iter_OD, 1);
                    if nDriverTemp>0
                        idxEndCol = idxEndCol + nDriverTemp;
                        if iter_OD==1
                            Z{iter_company}(2+n_path*(iter_time-1):end, 1:nDriverTemp) = 0;
                        elseif iter_OD == n_OD
                            Z{iter_company}(1+n_path*(iter_time-1):idxEndRow1, idxStartCol:end) = 0;
                        else
                            Z{iter_company}(1+n_path*(iter_time-1):idxEndRow1, idxStartCol:idxEndCol) = 0;
                            Z{iter_company}(idxEndRow1+nPathTemp+1:end, idxStartCol:idxEndCol) = 0;
                        end
                        idxStartCol = idxEndCol + 1;
                    end
                    idxEndRow1 = idxEndRow1 + nPathTemp;
                end
            end
        end
        time_array(iter, 8) = toc(ZStartTime);
        
        beta_tildaStartTime = tic;
        beta_tilda = -(1/rho)*lambda9 - c_tilda'*one_tilda + budget;
        beta_tilda(beta_tilda<0) = 0;
        time_array(iter, 9) = toc(beta_tildaStartTime);
    else
        % Block 2
        lambda1_vectorized = [];
        lambda3_vectorized = [];
        SOne_vectorized = [];
        q_company_vectorized = [];
        for iter_company=1:n_companies_ADMM
            lambda1_vectorized = [lambda1_vectorized; lambda1{iter_company}];
            lambda3_vectorized = [lambda3_vectorized; lambda3{iter_company}];
            SOne_vectorized = [SOne_vectorized; S{iter_company}*ones(size(S{iter_company}, 2), 1)];
            q_company_vectorized = [q_company_vectorized; q_company{iter_company}];
        end
        u_vectorizedStartTime = tic;
        u_vectorized = Tempu * (lambda1_vectorized + R_tilda'*lambda2 - ...
            D_tilda'*lambda3_vectorized - Delta_alpha*lambda7 + ...
            rho*SOne_vectorized - ...
            rho*R_tilda'*v_tilda_baseline + rho*R_tilda'*omega + ...
            rho*D_tilda'*q_company_vectorized + ...
            rho*(Delta_alpha)*(alpha.*gamma) + ...
            rho*(Delta_alpha)*eye_tilda*c_tilda);
        time_array(iter, 5) = toc(u_vectorizedStartTime);
        for iter_company=1:n_companies_ADMM
            u{iter_company} = u_vectorized(1+n_path*n_time*(iter_company-1):n_path*n_time*iter_company, 1);
        end
        
        for iter_company=1:n_companies_ADMM
            WStartTime = tic;
            tempRHS = b{iter_company}.*(B{iter_company}*etaDet);
            idxStart = 1;
            idxEnd = n_driver_company_time{iter_company, 1};
            for iter_time=1:n_time
                W{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) = ...
                    TempW_time*(-ones(n_path, 1)*lambda4{iter_company}(idxStart:idxEnd, 1)' + ...
                    lambda5{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) + ...
                    rho*ones(size(W{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd))) + ...
                    rho*S{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd));
                time_array(iter, 6) = time_array(iter, 6) + toc(WStartTime);
                
                HStartTime = tic;
                H{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) = ...
                    TempH_time{iter_time, 1}*(-delta_p_t{iter_time, 1}*lambda6{iter_company}(idxStart:idxEnd, 1)' + ...
                    lambda8{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) - ...
                    rho*delta_p_t{iter_time, 1}*beta{iter_company}(idxStart:idxEnd, 1)' + ...
                    rho*(delta_p_t{iter_time, 1}*(tempRHS(idxStart:idxEnd, 1))') + ...
                    rho*S{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd));
                time_array(iter, 7) = time_array(iter, 7) + toc(HStartTime);
                if iter_time~=n_time
                    idxStart = idxEnd + 1;
                    idxEnd = idxEnd + n_driver_company_time{iter_company, iter_time+1};
                end
            end
            if iter_company~=n_companies_ADMM
                idxStart = idxEnd + 1;
                idxEnd = idxEnd + n_driver_company_time{iter_company+1, 1};
            end
        end
        
        if iter<=n_iter_ADMM/6
            lambda_Z = lambda_Z_list(1);
        elseif  (iter>n_iter_ADMM/30) && (2*n_iter_ADMM/30)
            lambda_Z = lambda_Z_list(2);
        elseif (2*iter>n_iter_ADMM/30) && (4*n_iter_ADMM/30)
            lambda_Z = lambda_Z_list(3);
        else
            lambda_Z = lambda_Z_list(4);
        end
        ZStartTime = tic;
        for iter_company=1:n_companies_ADMM
            Z{iter_company} = (1/(rho-lambda_Z))*(lambda10{iter_company}+ ...
                rho*S{iter_company} - lambda_Z/2);
            if rho>lambda_Z
                Z{iter_company}(Z{iter_company}<0) = 0;
                Z{iter_company}(Z{iter_company}>1) = 1;
            else
                Z{iter_company}(Z{iter_company}<=0.5) = 1;
                Z{iter_company}(Z{iter_company}>0.5) = 0;
            end
            for iter_time=1:n_time
                if iter_time == 1
                    Z{iter_company}(1:n_path, sum(q_company{iter_company}(1:n_OD))+1:end) = 0;
                elseif iter_time == n_time
                    Z{iter_company}(n_path*(n_time-1)+1:end, 1:sum(q_company{iter_company}(1:n_OD*(n_time-1)))) = 0;
                else
                    Z{iter_company}(n_path*(iter_time-1)+1:n_path*iter_time, 1:sum(q_company{iter_company}(1:(iter_time-1)*n_OD))) = 0;
                    Z{iter_company}(n_path*(iter_time-1)+1:n_path*iter_time, sum(q_company{iter_company}(1:iter_time*n_OD))+1:end) = 0;
                end
            end
        end
        
        for iter_company=1:n_companies_ADMM
            idxEndRow1 = 0;
            idxStartCol = 1;
            idxEndCol = 0;
            for iter_time=1:n_time
                for iter_OD=1:n_OD
                    nPathTemp = num_path_v(iter_OD, 1);
                    nDriverTemp  = q_company_time{iter_company, iter_time}(iter_OD, 1);
                    if nDriverTemp>0
                        idxEndCol = idxEndCol + nDriverTemp;
                        if iter_OD==1
                            Z{iter_company}(2+n_path*(iter_time-1):end, 1:nDriverTemp) = 0;
                        elseif iter_OD == n_OD
                            Z{iter_company}(1+n_path*(iter_time-1):idxEndRow1, idxStartCol:end) = 0;
                        else
                            Z{iter_company}(1+n_path*(iter_time-1):idxEndRow1, idxStartCol:idxEndCol) = 0;
                            Z{iter_company}(idxEndRow1+nPathTemp+1:end, idxStartCol:idxEndCol) = 0;
                        end
                        idxStartCol = idxEndCol + 1;
                    end
                    idxEndRow1 = idxEndRow1 + nPathTemp;
                end
            end
        end
        time_array(iter, 8) = toc(ZStartTime);
        
        beta_tildaStartTime = tic;
        beta_tilda = -(1/rho)*lambda9 - c_tilda'*one_tilda + budget;
        beta_tilda(beta_tilda<0) = 0;
        time_array(iter, 9) = toc(beta_tildaStartTime);
        
        % Block 1
        uTemp = zeros(n_path*n_time, 1);
        for iter_company=1:n_companies_ADMM
            uTemp = uTemp + u{iter_company};
        end
        F_omega_total = 0;
        if no_obj
            uStartTime = tic;
            omega = -(1/rho)*lambda2 + R_matrix*uTemp + v_tilda_baseline;
            time_array(iter, 1) = toc(uStartTime);
            for iter_omega = 1:size(omega, 1)
                tt0Temp = tt0_array(link_loc(mod(iter_omega-1, size(tt0_array, 1)) + 1, 1), 3);
                wTemp = w_array(link_loc(mod(iter_omega-1, size(w_array, 1)) + 1, 1), 3);
                F_omega_total = F_omega_total + F_omega(omega(iter_omega, 1), tt0Temp, wTemp);
            end
        else
            for iter_omega = 1:size(omega, 1)
                tt0Temp = tt0_array(link_loc(mod(iter_omega-1, size(tt0_array, 1)) + 1, 1), 3);
                wTemp = w_array(link_loc(mod(iter_omega-1, size(w_array, 1)) + 1, 1), 3);
                r = R_matrix(iter_omega, :);
                l = lambda2(iter_omega);
                vTemp = v_tilda_baseline(iter_omega);
                %  FYI:  gradient_omega = @(omega, v, tt0, w, l, r, rho, u) tt0 + 5*(0.15*tt0/(w^4))*(omega^4) + l + rho*(omega-r*u-v);
                %  FYI:  bisectionMethodNew(gradient_omega, v, min_bis, max_bis, error_bisection, tt0, w, l, a, r, u)
                uStartTime = tic;
                omega(iter_omega) = bisectionMethodNew(gradient_omega, vTemp, min_bis, max_bis, error_bisection, tt0Temp, wTemp, l, r, rho, uTemp, maxIterBisection);
%                 omega(iter_omega)
                time_array(iter, 1) = time_array(iter, 1) + toc(uStartTime);
%                 time_array(iter, 1)
                % Our omega has constant volume in itself so v=0
                F_omega_total = F_omega_total + F_omega(omega(iter_omega, 1), tt0Temp, wTemp);
            end
        end
        total_travel_time_array(iter, 1) = F_omega_total/60;
        
        SStartTime = tic;
        if insertS && mod(iter, insertionStep)==0 && iter<=lastInsertion
            fprintf('Iteration %i >> S baseline injected', iter)
            for iter_company=1:n_companies_ADMM
                S{iter_company} = blkdiag(S_user_company_time{iter_company, :});
            end
        else
            for iter_company=1:n_companies_ADMM
                idxStart = 1;
                idxEnd = n_driver_company_time{iter_company, 1};
                for iter_time=1:n_time
                    S{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) =...
                        (-lambda1{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, 1)*ones(1,n_driver_company_time{iter_company, iter_time}) - ...
                        lambda5{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) - ...
                        lambda8{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) - ...
                        lambda10{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) + ....
                        rho*u{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, 1)*ones(1, n_driver_company_time{iter_company, iter_time}) + ...
                        rho*W{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) + ...
                        rho*H{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd) + ...
                        rho*Z{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd))*TempS_time{iter_company, iter_time};
                    if iter_time~=n_time
                        idxStart = idxEnd + 1;
                        idxEnd = idxEnd + n_driver_company_time{iter_company, iter_time+1};
                    end
                end
            end
        end
        time_array(iter, 2) = toc(SStartTime);
        betaStartTime = tic;
        for iter_company=1:n_companies_ADMM
            tempRHS = b{iter_company}.*(B{iter_company}*etaDet);
            idxStart = 1;
            idxEnd = n_driver_company_time{iter_company, 1};
            for iter_time=1:n_time
                beta{iter_company}(idxStart:idxEnd, 1) = ...
                    (1/rho) * (-lambda6{iter_company}(idxStart:idxEnd, 1) - ...
                    rho*H{iter_company}((iter_time-1)*n_path+1:iter_time*n_path, idxStart:idxEnd)'*delta_p_t{iter_time, 1} + ...
                    rho*tempRHS(idxStart:idxEnd, 1));
                if iter_time~=n_time
                    idxStart = idxEnd + 1;
                    idxEnd = idxEnd + n_driver_company_time{iter_company, iter_time+1};
                end
            end
            beta{iter_company}(beta{iter_company}<0) = 0;
        end
        time_array(iter, 3) = toc(betaStartTime);
        
        c_tildaStartTime = tic;
        c_tilda = Tempc_tilda * (eye_tilda'*lambda7 - lambda9*one_tilda - ...
            rho*((eye_tilda')*(alpha.*gamma)) + ...
            rho*((eye_tilda')*(alpha.*(Delta'*u_vectorized))) - ...
            rho*beta_tilda*one_tilda + rho*budget*one_tilda);
        c_tilda(c_tilda<0) = 0;
        time_array(iter, 4) = toc(c_tildaStartTime);
        
    end
    
    
    uTemp = zeros(n_path*n_time, 1);
    for iter_company=1:n_companies_ADMM
        uTemp = uTemp + u{iter_company};
    end
    dualStartTime = tic;
    for iter_company=1:n_companies_ADMM
        dualStartTime1 = tic;
        lambda1{iter_company} = lambda1{iter_company} + ...
            rho*(S{iter_company}*ones(n_driver_company{iter_company},1)-u{iter_company});
        time_array_dual(iter, 1) = toc(dualStartTime1);
        
        dualStartTime3 = tic;
        lambda3{iter_company} = lambda3{iter_company} + ...
            rho*(D*u{iter_company} - q_company{iter_company});
        time_array_dual(iter, 3) = toc(dualStartTime3);
        
        dualStartTime4 = tic;
        lambda4{iter_company} = lambda4{iter_company} + ...
            rho*(W{iter_company}'*ones(size(W{iter_company}, 1), 1) - ...
            ones(size(W{iter_company}, 2), 1));
        time_array_dual(iter, 4) = toc(dualStartTime4);
        
        dualStartTime5 = tic;
        lambda5{iter_company} = lambda5{iter_company} + ...
            rho*(S{iter_company} - W{iter_company});
        time_array_dual(iter, 5) = toc(dualStartTime5);
        
        dualStartTime6 = tic;
        lambda6{iter_company} = lambda6{iter_company} + ...
            rho*(H{iter_company}'*delta_pDet + beta{iter_company} - ...
            b{iter_company}.*(B{iter_company}*etaDet));
        time_array_dual(iter, 6) = toc(dualStartTime6);
        
        dualStartTime8 = tic;
        lambda8{iter_company} = lambda8{iter_company} + ...
            rho*(S{iter_company}-H{iter_company});
        time_array_dual(iter, 8) = toc(dualStartTime8);
        
        dualStartTime10 = tic;
        lambda10{iter_company} = lambda10{iter_company} + ...
            rho*(S{iter_company} - Z{iter_company});
        time_array_dual(iter, 10) = toc(dualStartTime10);
    end
    dualStartTime2 = tic;
    lambda2 = lambda2 + rho*(omega - R_matrix*uTemp - v_tilda_baseline);
    time_array_dual(iter, 2) = toc(dualStartTime2);
    dualStartTime7 = tic;
    lambda7 = lambda7 + rho*(alpha.*(Delta'*u_vectorized - gamma) - eye_tilda*c_tilda);
    time_array_dual(iter, 7) = toc(dualStartTime7);
    dualStartTime9 = tic;
    lambda9 = lambda9 + rho*(c_tilda'*one_tilda + beta_tilda - budget);
    time_array_dual(iter, 9) = toc(dualStartTime9);
    time_array(iter, 10) = toc(dualStartTime);
    
    ConstViolation_normalized(iter) = ConstViolation_normalized(iter) + ...
        norm(omega - R_matrix*uTemp - v_tilda_baseline)/(norm(omega) + norm(R_matrix, 'fro')*norm(uTemp) + norm(v_tilda_baseline)) + ...
        norm(alpha.*(Delta'*u_vectorized - gamma) - eye_tilda*c_tilda)/(norm(alpha)*(norm(Delta, 'fro')*norm(u_vectorized)+norm(gamma)) + norm(eye_tilda)*norm(c_tilda)) + ...
        norm(c_tilda'*one_tilda + beta_tilda - budget,'fro')/(norm(c_tilda)*norm(one_tilda) + norm(beta_tilda) + norm(budget));
    
    ConstViolation(iter) = norm(omega - R_matrix*uTemp - v_tilda_baseline) + ...
        norm(alpha.*(Delta'*u_vectorized - gamma) - eye_tilda*c_tilda) + ...
        norm(c_tilda'*one_tilda + beta_tilda - budget,'fro');
    
    Augmented_Lagrangian(iter) = F_omega_total + ...
        lambda2'*(omega - R_tilda*u_vectorized - v_tilda_baseline) + ...
        lambda7'*(Delta_alpha'*u_vectorized - alpha.*gamma - eye_tilda*c_tilda) + ...
        lambda9'*(c_tilda'*one_tilda + beta_tilda - budget) + ...
        (rho/2)*norm(omega - R_tilda*u_vectorized - v_tilda_baseline)^2 + ...
        (rho/2)*norm(Delta_alpha'*u_vectorized - alpha.*gamma - eye_tilda*c_tilda)^2 + ...
        (rho/2)*norm(c_tilda'*one_tilda + beta_tilda - budget)^2;
    
    for iter_company=1:n_companies_ADMM
        ConstViolation_normalized(iter) = ConstViolation_normalized(iter) + ...
            norm(S{iter_company}*ones(n_driver_company{iter_company},1)-u{iter_company})/(norm(S{iter_company})*norm(ones(n_driver_company{iter_company},1)) + norm(u{iter_company})) + ...
            norm(D*u{iter_company} - q_company{iter_company})/(norm(D, 'fro')*norm(u{iter_company}) + norm(q_company{iter_company})) + ...
            norm(W{iter_company}'*ones(size(W{iter_company}, 1), 1) - ones(size(W{iter_company}, 2), 1))/(norm(W{iter_company}, 'fro')*norm(ones(size(W{iter_company}, 1), 1)) + norm(ones(size(W{iter_company}, 2), 1))) + ...
            norm(S{iter_company}-W{iter_company},'fro')/(norm(W{iter_company}, 'fro') + norm(S{iter_company}, 'fro')) + ...
            norm(H{iter_company}'*delta_pDet + beta{iter_company} - b{iter_company}.*(B{iter_company}*etaDet))/(norm(H{iter_company}, 'fro')*norm(delta_pDet) + norm(beta{iter_company}) + norm(b{iter_company})*norm(B{iter_company}, 'fro')*norm(etaDet)) +...
            norm(S{iter_company}-H{iter_company},'fro')/(norm(S{iter_company}, 'fro') + norm(H{iter_company}, 'fro')) + ...
            norm(S{iter_company}-Z{iter_company},'fro')/(norm(S{iter_company}, 'fro') + norm(Z{iter_company}, 'fro'));
        
        ConstViolation(iter) = norm(S{iter_company}*ones(n_driver_company{iter_company},1)-u{iter_company}) + ...
            norm(D*u{iter_company} - q_company{iter_company}) + ...
            norm(W{iter_company}'*ones(size(W{iter_company}, 1), 1) - ones(size(W{iter_company}, 2), 1)) + ...
            norm(S{iter_company}-W{iter_company},'fro') + ...
            norm(H{iter_company}'*delta_pDet + beta{iter_company} - b{iter_company}.*(B{iter_company}*etaDet)) +...
            norm(S{iter_company}-H{iter_company},'fro') + ...
            norm(S{iter_company}-Z{iter_company},'fro');
        
        Augmented_Lagrangian(iter) = lambda1{iter_company}'*(S{iter_company}*ones(n_driver_company{iter_company},1)-u{iter_company}) + ...
            lambda3{iter_company}'*(D*u{iter_company} - q_company{iter_company}) + ...
            lambda4{iter_company}'*(W{iter_company}'*ones(size(W{iter_company}, 1), 1) - ones(size(W{iter_company}, 2), 1)) + ...
            norm(lambda5{iter_company}'*(S{iter_company} - W{iter_company}), 'fro') + ...
            lambda6{iter_company}'*(H{iter_company}'*delta_pDet + beta{iter_company} - b{iter_company}.*(B{iter_company})*etaDet) + ...
            norm(lambda8{iter_company}'*(S{iter_company} - H{iter_company}), 'fro') + ...
            norm(lambda10{iter_company}'*(S{iter_company}-Z{iter_company}),'fro') + ...
            (rho/2)*norm(S{iter_company}*ones(n_driver_company{iter_company},1)-u{iter_company})^2 + ...
            (rho/2)*norm(D*u{iter_company} - q_company{iter_company})^2 + ...
            (rho/2)*norm(W{iter_company}'*ones(size(W{iter_company}, 1), 1) - ones(size(W{iter_company}, 2), 1))^2 + ...
            (rho/2)*norm((S{iter_company} - W{iter_company}), 'fro')^2 + ...
            (rho/2)*norm(H{iter_company}'*delta_pDet + beta{iter_company} - b{iter_company}.*(B{iter_company})*etaDet)^2 + ...
            (rho/2)*norm((S{iter_company} - H{iter_company}), 'fro')^2 + ...
            (rho/2)*norm((S{iter_company}-Z{iter_company}),'fro')^2;
    end
    
    if mod(iter, normWindow)==0
        fprintf('Iteration %i.\n', iter)
        if printNorm
            fprintf('Norm of omega: %.10f\n', norm(omega))
            for iter_company=1:n_companies_ADMM
                fprintf('Fro Norm of S%i: %.10f\n', iter_company, norm(S{iter_company}, 'fro'))
                fprintf('Norm of beta%i: %.10f\n', iter_company, norm(beta{iter_company}))
                fprintf('Norm of c%i: %.10f\n', iter_company, norm(c_tilda(iter_company)))
                fprintf('Norm of u%i: %.10f\n', iter_company, norm(u{iter_company}))
                fprintf('Fro Norm of W%i: %.10f\n', iter_company, norm(W{iter_company}, 'fro'))
                fprintf('Fro Norm of H%i: %.10f\n', iter_company, norm(H{iter_company}, 'fro'))
                fprintf('Fro Norm of Z%i: %.10f\n', iter_company, norm(Z{iter_company}, 'fro'))
                fprintf('Norm of beta tilda%i: %.10f\n', iter_company, beta_tilda)
                
                fprintf('Norm of Gap1%i (normalized): %.10f\n', iter_company, norm(S{iter_company}*ones(n_driver_company{iter_company},1)-u{iter_company})/(norm(S{iter_company})*norm(ones(n_driver_company{iter_company},1)) + norm(u{iter_company})))
                fprintf('Norm of Gap3%i (normalized): %.10f\n', iter_company, norm(D*u{iter_company} - q_company{iter_company})/(norm(D, 'fro')*norm(u{iter_company}) + norm(q_company{iter_company})))
                fprintf('Norm of Gap4%i (normalized): %.10f\n', iter_company, norm(W{iter_company}'*ones(size(W{iter_company}, 1), 1) - ones(size(W{iter_company}, 2), 1))/(norm(W{iter_company}, 'fro')*norm(ones(size(W{iter_company}, 1), 1)) + norm(ones(size(W{iter_company}, 2), 1))))
                fprintf('Norm of Gap5%i (normalized): %.10f\n', iter_company, norm(S{iter_company}-W{iter_company},'fro')/(norm(W{iter_company}, 'fro') + norm(S{iter_company}, 'fro')))
                fprintf('Norm of Gap6%i (normalized): %.10f\n', iter_company, norm(H{iter_company}'*delta_pDet + beta{iter_company} - b{iter_company}.*(B{iter_company}*etaDet))/(norm(H{iter_company}, 'fro')*norm(delta_pDet) + norm(beta{iter_company}) + norm(b{iter_company})*norm(B{iter_company}, 'fro')*norm(etaDet)))
                fprintf('Norm of Gap8%i (normalized): %.10f\n', iter_company, norm(S{iter_company}-H{iter_company},'fro')/(norm(S{iter_company}, 'fro') + norm(H{iter_company}, 'fro')))
                fprintf('Norm of Gap10_%i (normalized): %.10f\n', iter_company, norm(S{iter_company}-Z{iter_company},'fro')/(norm(S{iter_company}, 'fro') + norm(Z{iter_company}, 'fro')))
            end
            fprintf('Norm of Gap2 (normalized): %.10f\n', norm(omega - R_matrix*uTemp - v_tilda_baseline)/(norm(omega) + norm(R_matrix, 'fro')*norm(uTemp) + norm(v_tilda_baseline)))
            fprintf('Norm of Gap7 (normalized): %.10f\n', norm(alpha.*(Delta'*u_vectorized - gamma) - eye_tilda*c_tilda)/(norm(alpha)*(norm(Delta, 'fro')*norm(u_vectorized)+norm(gamma)) + norm(eye_tilda)*norm(c_tilda)))
            fprintf('Norm of Gap9 (normalized): %.10f\n', norm(c_tilda'*one_tilda + beta_tilda - budget,'fro')/(norm(c_tilda)*norm(one_tilda) + norm(beta_tilda) + norm(budget)))
            
            fprintf('Gap with normalization: %.10f\n', ConstViolation_normalized(iter))
            fprintf('Gap without normalization: %.10f\n', ConstViolation(iter))
            
            fprintf('Computation time of omega: %.6f seconds\n', mean(time_array(1:iter, 1)))
            fprintf('Computation time of S: %.6f seconds\n', mean(time_array(1:iter, 2)))
            fprintf('Computation time of beta: %.6f seconds\n', mean(time_array(1:iter, 3)))
            fprintf('Computation time of c: %.6f seconds\n', mean(time_array(1:iter, 4)))
            fprintf('Computation time of u: %.6f seconds\n', mean(time_array(1:iter, 5)))
            fprintf('Computation time of W: %.6f seconds\n', mean(time_array(1:iter, 6)))
            fprintf('Computation time of H: %.6f seconds\n', mean(time_array(1:iter, 7)))
            fprintf('Computation time of Z: %.6f seconds\n', mean(time_array(1:iter, 8)))
            fprintf('Computation time of beta tilda: %.6f seconds\n', mean(time_array(1:iter, 9)))
            fprintf('Computation time of dual variables: %.6f seconds\n', mean(time_array(1:iter, 10)))
            fprintf('Computation time of dual variable 1: %.6f seconds\n', mean(time_array_dual(1:iter, 1)))
            fprintf('Computation time of dual variable 2: %.6f seconds\n', mean(time_array_dual(1:iter, 2)))
            fprintf('Computation time of dual variable 3: %.6f seconds\n', mean(time_array_dual(1:iter, 3)))
            fprintf('Computation time of dual variable 4: %.6f seconds\n', mean(time_array_dual(1:iter, 4)))
            fprintf('Computation time of dual variable 5: %.6f seconds\n', mean(time_array_dual(1:iter, 5)))
            fprintf('Computation time of dual variable 6: %.6f seconds\n', mean(time_array_dual(1:iter, 6)))
            fprintf('Computation time of dual variable 7: %.6f seconds\n', mean(time_array_dual(1:iter, 7)))
            fprintf('Computation time of dual variable 8: %.6f seconds\n', mean(time_array_dual(1:iter, 8)))
            fprintf('Computation time of dual variable 9: %.6f seconds\n', mean(time_array_dual(1:iter, 9)))
            fprintf('Computation time of dual variable 10: %.6f seconds\n', mean(time_array_dual(1:iter, 10)))
        end
    end
    %% LP Approximation
    time_LP_start = tic;
    if iter>=n_iter_ADMM-LPWindow && mod(iter, LPStep)==0

        fprintf('LP @ iteration %i\n', iter)
        cvx_solver gurobi
        cvx_solver_settings('MIPGap', .01);
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
%         idxEndRow1 = 0;
%         idxStartCol = cell(n_companies_ADMM, n_time);
%         idxStartCol(:) = {1};
%         idxEndCol = cell(n_companies_ADMM, n_time);
%         idxEndCol(:) = {0};
%         nDriverTemp = cell(n_companies_ADMM, n_time);
%         for iter_OD=1:n_OD
%             nPathTemp = num_path_v(iter_OD, 1);
%             for iter_company=1:n_companies_ADMM
%                 for iter_time=1:n_time
%                     nDriverTemp{iter_company, iter_time}  = q_company_time{iter_company, iter_time}(iter_OD, 1);
%                 end
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+0}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+0} = idxEndCol{1, n_time_inc_start_ADMM+0} + nDriverTemp{1, n_time_inc_start_ADMM+0};
%                 if iter_OD==1
%                     S11_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+0}) == 0;
%                 elseif iter_OD == n_OD
%                     S11_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+0}:end) == 0;
%                 else
%                     S11_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+0}:idxEndCol{1, n_time_inc_start_ADMM+0}) == 0;
%                     S11_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+0}:idxEndCol{1, n_time_inc_start_ADMM+0}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+0} = idxEndCol{1, n_time_inc_start_ADMM+0} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+1}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+1} = idxEndCol{1, n_time_inc_start_ADMM+1} + nDriverTemp{1, n_time_inc_start_ADMM+1};
%                 if iter_OD==1
%                     S12_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+1}) == 0;
%                 elseif iter_OD == n_OD
%                     S12_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+1}:end) == 0;
%                 else
%                     S12_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+1}:idxEndCol{1, n_time_inc_start_ADMM+1}) == 0;
%                     S12_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+1}:idxEndCol{1, n_time_inc_start_ADMM+1}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+1} = idxEndCol{1, n_time_inc_start_ADMM+1} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+2}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+2} = idxEndCol{1, n_time_inc_start_ADMM+2} + nDriverTemp{1, n_time_inc_start_ADMM+2};
%                 if iter_OD==1
%                     S13_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+2}) == 0;
%                 elseif iter_OD == n_OD
%                     S13_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+2}:end) == 0;
%                 else
%                     S13_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+2}:idxEndCol{1, n_time_inc_start_ADMM+2}) == 0;
%                     S13_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+2}:idxEndCol{1, n_time_inc_start_ADMM+2}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+2} = idxEndCol{1, n_time_inc_start_ADMM+2} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+3}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+3} = idxEndCol{1, n_time_inc_start_ADMM+3} + nDriverTemp{1, n_time_inc_start_ADMM+3};
%                 if iter_OD==1
%                     S14_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+3}) == 0;
%                 elseif iter_OD == n_OD
%                     S14_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+3}:end) == 0;
%                 else
%                     S14_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+3}:idxEndCol{1, n_time_inc_start_ADMM+3}) == 0;
%                     S14_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+3}:idxEndCol{1, n_time_inc_start_ADMM+3}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+3} = idxEndCol{1, n_time_inc_start_ADMM+3} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+4}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+4} = idxEndCol{1, n_time_inc_start_ADMM+4} + nDriverTemp{1, n_time_inc_start_ADMM+4};
%                 if iter_OD==1
%                     S15_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+4}) == 0;
%                 elseif iter_OD == n_OD
%                     S15_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+4}:end) == 0;
%                 else
%                     S15_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+4}:idxEndCol{1, n_time_inc_start_ADMM+4}) == 0;
%                     S15_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+4}:idxEndCol{1, n_time_inc_start_ADMM+4}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+4} = idxEndCol{1, n_time_inc_start_ADMM+4} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+5}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+5} = idxEndCol{1, n_time_inc_start_ADMM+5} + nDriverTemp{1, n_time_inc_start_ADMM+5};
%                 if iter_OD==1
%                     S16_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+5}) == 0;
%                 elseif iter_OD == n_OD
%                     S16_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+5}:end) == 0;
%                 else
%                     S16_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+5}:idxEndCol{1, n_time_inc_start_ADMM+5}) == 0;
%                     S16_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+5}:idxEndCol{1, n_time_inc_start_ADMM+5}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+5} = idxEndCol{1, n_time_inc_start_ADMM+5} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+6}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+6} = idxEndCol{1, n_time_inc_start_ADMM+6} + nDriverTemp{1, n_time_inc_start_ADMM+6};
%                 if iter_OD==1
%                     S17_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+6}) == 0;
%                 elseif iter_OD == n_OD
%                     S17_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+6}:end) == 0;
%                 else
%                     S17_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+6}:idxEndCol{1, n_time_inc_start_ADMM+6}) == 0;
%                     S17_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+6}:idxEndCol{1, n_time_inc_start_ADMM+6}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+6} = idxEndCol{1, n_time_inc_start_ADMM+6} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+7}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+7} = idxEndCol{1, n_time_inc_start_ADMM+7} + nDriverTemp{1, n_time_inc_start_ADMM+7};
%                 if iter_OD==1
%                     S18_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+7}) == 0;
%                 elseif iter_OD == n_OD
%                     S18_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+7}:end) == 0;
%                 else
%                     S18_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+7}:idxEndCol{1, n_time_inc_start_ADMM+7}) == 0;
%                     S18_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+7}:idxEndCol{1, n_time_inc_start_ADMM+7}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+7} = idxEndCol{1, n_time_inc_start_ADMM+7} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+8}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+8} = idxEndCol{1, n_time_inc_start_ADMM+8} + nDriverTemp{1, n_time_inc_start_ADMM+8};
%                 if iter_OD==1
%                     S19_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+8}) == 0;
%                 elseif iter_OD == n_OD
%                     S19_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+8}:end) == 0;
%                 else
%                     S19_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+8}:idxEndCol{1, n_time_inc_start_ADMM+8}) == 0;
%                     S19_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+8}:idxEndCol{1, n_time_inc_start_ADMM+8}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+8} = idxEndCol{1, n_time_inc_start_ADMM+8} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+9}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+9} = idxEndCol{1, n_time_inc_start_ADMM+9} + nDriverTemp{1, n_time_inc_start_ADMM+9};
%                 if iter_OD==1
%                     S110_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+9}) == 0;
%                 elseif iter_OD == n_OD
%                     S110_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+9}:end) == 0;
%                 else
%                     S110_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+9}:idxEndCol{1, n_time_inc_start_ADMM+9}) == 0;
%                     S110_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+9}:idxEndCol{1, n_time_inc_start_ADMM+9}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+9} = idxEndCol{1, n_time_inc_start_ADMM+9} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+10}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+10} = idxEndCol{1, n_time_inc_start_ADMM+10} + nDriverTemp{1, n_time_inc_start_ADMM+10};
%                 if iter_OD==1
%                     S111_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+10}) == 0;
%                 elseif iter_OD == n_OD
%                     S111_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+10}:end) == 0;
%                 else
%                     S111_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+10}:idxEndCol{1, n_time_inc_start_ADMM+10}) == 0;
%                     S111_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+10}:idxEndCol{1, n_time_inc_start_ADMM+10}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+10} = idxEndCol{1, n_time_inc_start_ADMM+10} + 1;
%             end
%             if nDriverTemp{1, n_time_inc_start_ADMM+11}>0
%                 idxEndCol{1, n_time_inc_start_ADMM+11} = idxEndCol{1, n_time_inc_start_ADMM+11} + nDriverTemp{1, n_time_inc_start_ADMM+11};
%                 if iter_OD==1
%                     S112_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_ADMM+11}) == 0;
%                 elseif iter_OD == n_OD
%                     S112_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+11}:end) == 0;
%                 else
%                     S112_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_ADMM+11}:idxEndCol{1, n_time_inc_start_ADMM+11}) == 0;
%                     S112_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_ADMM+11}:idxEndCol{1, n_time_inc_start_ADMM+11}) == 0;
%                 end
%                 idxStartCol{1, n_time_inc_start_ADMM+11} = idxEndCol{1, n_time_inc_start_ADMM+11} + 1;
%             end
%             idxEndRow1 = idxEndRow1 + nPathTemp;
%         end
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
        totalTimeLP = toc(time_LP_start);
        
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

        %% Save arrays
%         folderRun = fullfile(inputFolder0, ...
%             strcat('Det_initAll2_MultT', ...
%             '_b', num2str(budget), ...
%             '_sD', num2str(seedData), ...
%             '_sA', num2str(seedADMM), ...
%             '_r', num2str(rho), ...
%             '_it', num2str(n_iter_ADMM),...
%             '_VOT', num2str(VOT), ...
%             '_nC', num2str(n_companies_ADMM), ...
%             '_f', fairness, ...
%             '_percC', setting_perc_companies, ...
%             '_initSB_', LogicalStr{initializeSBaseline + 1}, ...
%             '_percNonU', setting_output_ADMM, ...
%             '_nTIS', num2str(n_time_inc_start_ADMM), '_nTIE', num2str(n_time_inc_end_ADMM), ...
%             '_itN', num2str(iterRun)));
%         mkdir(folderRun);
        fileName2 = fullfile(folderRun,...
            strcat('result_', num2str(iterRun), '.mat'));
        save(fileName2);
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
        S_binary_full_blkdiag{1, 1} = round(blkdiag(S_check_response{1, :}));
        S_binary_full_blkdiag{1, 1} = transpose(sortrows(transpose(S_binary_full_blkdiag{1, 1}),'descend'));

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
        S_binary_full_blkdiag = cell(n_companies_ADMM, 1);
        S_binary_full_blkdiag{1, 1} = S_binary_full_blkdiag{1, 1};
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
%         % Deviation from min tt
%         dev1 = gap_baseline1(gap_baseline1>0);
%         dev2 = gap_baseline2(gap_baseline2>0);
%         fprintf('Mean of deviation from min tt of company 1: %.6f\n', mean(dev1))
%         fprintf('Mean of deviation from min tt of company 2: %.6f\n', mean(dev2))
%         fprintf('Mean of deviation from min tt of companies: %.6f\n', mean([dev1; dev2]))
%         fprintf('Max of deviation from min tt of company 1: %.6f\n', max(dev1))
%         fprintf('Max of deviation from min tt of company 2: %.6f\n', max(dev2))
%         fprintf('Max of deviation from min tt of companies: %.6f\n', max([dev1; dev2]))
%         fprintf('Std of deviation from min tt of company 1: %.6f\n', std(dev1))
%         fprintf('Std of deviation from min tt of company 2: %.6f\n', std(dev2))
%         fprintf('Std of deviation from min tt of companies: %.6f\n', std([dev1; dev2]))
        
        %% Save
        %         if saveLastIters == true
        %             if lambda_Z~=0
        %                 outputFolder = fullfile('data', region_, setting_region, strcat('7AM_reg_', no_obj_str, '_r', num2str(rho), '_It', num2str(n_iter_ADMM), '_b', num2str(budget), '_VOT', num2str(VOT), '_nC', num2str(n_companies_ADMM), '_percC', setting_perc_companies, '_nonuser', setting_output_ADMM, '_f', fairness, '_initSB_', LogicalStr{initializeSBaseline + 1}));
        %             else
        %                 outputFolder = fullfile('data', region_, setting_region, strcat('7AM_', no_obj_str, '_r', num2str(rho), '_It', num2str(n_iter_ADMM), '_b', num2str(budget), '_VOT', num2str(VOT), '_nC', num2str(n_companies_ADMM), '_percC', setting_perc_companies, '_nonuser', setting_output_ADMM, '_f', fairness, '_initSB_', LogicalStr{initializeSBaseline + 1}));
        %             end
        %             mkdir(outputFolder);
        %             filenameOutput = fullfile(outputFolder, 'AllVarOutput_Linear.mat');
        %             save(filenameOutput)
        %         end
        counter_save = counter_save + 1;
    end
    %     end
end
fprintf('Computation time of omega: %.6f seconds\n', sum(time_array(1:iter, 1))/iter)
fprintf('Computation time of S: %.6f seconds\n', sum(time_array(1:iter, 2))/iter)
fprintf('Computation time of beta: %.6f seconds\n', sum(time_array(1:iter, 3))/iter)
fprintf('Computation time of c: %.6f seconds\n', sum(time_array(1:iter, 4))/iter)
fprintf('Computation time of u: %.6f seconds\n', sum(time_array(1:iter, 5))/iter)
fprintf('Computation time of W: %.6f seconds\n', sum(time_array(1:iter, 6))/iter)
fprintf('Computation time of H: %.6f seconds\n', sum(time_array(1:iter, 7))/iter)
fprintf('Computation time of Z: %.6f seconds\n', sum(time_array(1:iter, 8))/iter)
fprintf('Computation time of beta tilda: %.6f seconds\n', sum(time_array(1:iter, 9))/iter)
fprintf('Computation time of dual variables: %.6f seconds\n', sum(time_array(1:iter, 10))/iter)
totalTimeRun = toc(startTime0);
fprintf('Total run time LP: %.6f minutes\n', totalTimeLP/60)
fprintf('Total run time: %.6f minutes\n', totalTimeRun/60)

%% Keep record of runtime
% Example runtime data
avg_time_array = mean(time_array, 1);  
avg_time_array_dual = mean(time_array_dual, 1);  
avg_time_array_complete = [avg_time_array, avg_time_array_dual];

% Define the file name
fileNameRuntime = fullfile(folderRun, 'runtimes.csv');
% Check if the file exists
if isfile(fileNameRuntime)
    % File exists, load existing data
    existingRuntime = readmatrix(fileNameRuntime);
    % Append new data
    updatedRuntime = [existingRuntime; avg_time_array_complete];
else
    % File does not exist, initialize with newData
    updatedRuntime = avg_time_array_complete;
end

% Save updated data to CSV
writematrix(updatedRuntime, fileNameRuntime);

%% Save arrays

% folderRun = fullfile(inputFolder0, ...
%     strcat('Det_initAll2_MultT', ...
%     '_b', num2str(budget), ...
%     '_sD', num2str(seedData), ...
%     '_sA', num2str(seedADMM), ...
%     '_r', num2str(rho), ...
%     '_it', num2str(n_iter_ADMM),...
%     '_VOT', num2str(VOT), ...
%     '_nC', num2str(n_companies_ADMM), ...
%     '_f', fairness, ...
%     '_percC', setting_perc_companies, ...
%     '_initSB_', LogicalStr{initializeSBaseline + 1}, ...
%     '_percNonU', setting_output_ADMM, ...
%     '_nTIS', num2str(n_time_inc_start_ADMM), '_nTIE', num2str(n_time_inc_end_ADMM), ...
%     '_itN', num2str(iterRun)));
% 
% mkdir(folderRun);
% Remove large variables with no information
clearvars TempS Tempu TempW TempH TempS_time TempW_time TempH_time
fileName2 = fullfile(folderRun,...
    strcat('result_', num2str(iterRun), '.mat'));
save(fileName2);
%% Plot
% Convergence plots
figure()
subplot(2, 2, 1)
plot(ConstViolation_normalized)
title('Normalized gap')
subplot(2, 2, 2)
plot(ConstViolation)
title('Original gap')
subplot(2, 2, 3)
plot(Augmented_Lagrangian)
title('Augmented lagrangian')
subplot(2, 2, 4)
plot(total_travel_time_array)
title('Total travel time (hours)')
filenamePlotPNG = fullfile(folderRun, 'plot.png');
saveas(gcf,filenamePlotPNG)
% filenamePlotPNG = fullfile(folderRun, 'plot.fig');
% saveas(gcf,filenamePlotPNG)

%% Variable sizes
% Check memory usage
variable_info = whos;
for i = 1:length(variable_info)
variable_info(i).memory_MB = variable_info(i).bytes / (1024^2); % Convert bytes to MB
end
% Sort variables based on memory usage
[~, sorted_indices] = sort([variable_info.memory_MB], 'descend');
sorted_variable_info = variable_info(sorted_indices);
% Display sorted memory usage
fprintf('Variable Name\tMemory (MB)\n');
for i = 1:length(sorted_variable_info)
fprintf('%s\t\t%.2f\n', sorted_variable_info(i).name, sorted_variable_info(i).memory_MB);
end

%% END
% Beep sound
sound(sin(1:3000));
diary off; % Stop logging
% Python script location
pythonScript = fullfile('../lib', 'convert_to_pdf.py');
% Call the Python script
% system(sprintf('python "%s"', pythonScript));
system(sprintf('python3 "%s" "%s"', pythonScript, log_file));
