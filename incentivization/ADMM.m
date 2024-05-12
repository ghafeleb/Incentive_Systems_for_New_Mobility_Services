function ADMM(iterRun, nonuser_perc_prob, budget, n_iter_ADMM, ...
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
%% ADMM
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
    
    
%     normStartTime = tic;
%     norm_variables(iter, 1) = norm(omega);
%     for iter_company=1:n_companies_ADMM
%         norm_variables(iter, 2) = norm(S{iter_company}, 'fro');
%         norm_variables(iter, 3) = norm(beta{iter_company});
%         norm_variables(iter, 4) = norm(c_tilda(iter_company));
%         norm_variables(iter, 5) = norm(u{iter_company});
%         norm_variables(iter, 6) = norm(W{iter_company}, 'fro');
%         norm_variables(iter, 7) = norm(H{iter_company}, 'fro');
%         norm_variables(iter, 8) = norm(Z{iter_company}, 'fro');
%         norm_variables(iter, 9) = beta_tilda;
% 
%         norm_variables(iter, 10) = norm(S{iter_company}*ones(n_driver_company{iter_company},1)-u{iter_company})/(norm(S{iter_company})*norm(ones(n_driver_company{iter_company},1)) + norm(u{iter_company}));
%         norm_variables(iter, 11) = norm(D*u{iter_company} - q_company{iter_company})/(norm(D, 'fro')*norm(u{iter_company}) + norm(q_company{iter_company}));
%         norm_variables(iter, 12) = norm(W{iter_company}'*ones(size(W{iter_company}, 1), 1) - ones(size(W{iter_company}, 2), 1))/(norm(W{iter_company}, 'fro')*norm(ones(size(W{iter_company}, 1), 1)) + norm(ones(size(W{iter_company}, 2), 1)));
%         norm_variables(iter, 13) = norm(S{iter_company}-W{iter_company},'fro')/(norm(W{iter_company}, 'fro') + norm(S{iter_company}, 'fro'));
%         norm_variables(iter, 14) = norm(H{iter_company}'*delta_pDet + beta{iter_company} - b{iter_company}.*(B{iter_company}*etaDet))/(norm(H{iter_company}, 'fro')*norm(delta_pDet) + norm(beta{iter_company}) + norm(b{iter_company})*norm(B{iter_company}, 'fro')*norm(etaDet));
%         norm_variables(iter, 15) = norm(S{iter_company}-H{iter_company},'fro')/(norm(S{iter_company}, 'fro') + norm(H{iter_company}, 'fro'));
%         norm_variables(iter, 16) = norm(S{iter_company}-Z{iter_company},'fro')/(norm(S{iter_company}, 'fro') + norm(Z{iter_company}, 'fro'));
%     end
%     norm_variables(iter, 17) = norm(omega - R_matrix*uTemp - v_tilda_baseline)/(norm(omega) + norm(R_matrix, 'fro')*norm(uTemp) + norm(v_tilda_baseline));
%     norm_variables(iter, 18) = norm(alpha.*(Delta'*u_vectorized - gamma) - eye_tilda*c_tilda)/(norm(alpha)*(norm(Delta, 'fro')*norm(u_vectorized)+norm(gamma)) + norm(eye_tilda)*norm(c_tilda));
%     norm_variables(iter, 19) = norm(c_tilda'*one_tilda + beta_tilda - budget,'fro')/(norm(c_tilda)*norm(one_tilda) + norm(beta_tilda) + norm(budget));
%     normTime = toc(normStartTime)

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
end

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
totalTimeRun = toc(startTime0);
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
% Save plot
filenamePlotPNG = fullfile(folderRun, 'plot.png');
saveas(gcf,filenamePlotPNG)
% filenamePlotPNG = fullfile(folderRun, 'plot.fig');
% saveas(gcf,filenamePlotPNG)

% %% Variable sizes
% % Check memory usage
% variable_info = whos;
% for i = 1:length(variable_info)
% variable_info(i).memory_MB = variable_info(i).bytes / (1024^2); % Convert bytes to MB
% end
% % Sort variables based on memory usage
% [~, sorted_indices] = sort([variable_info.memory_MB], 'descend');
% sorted_variable_info = variable_info(sorted_indices);
% % Display sorted memory usage
% fprintf('Variable Name\tMemory (MB)\n');
% for i = 1:length(sorted_variable_info)
% fprintf('%s\t\t%.2f\n', sorted_variable_info(i).name, sorted_variable_info(i).memory_MB);
% end

%% END
% Beep sound
sound(sin(1:3000));
diary off; % Stop logging
% Python script location
pythonScript = fullfile('../lib', 'convert_to_pdf.py');
% Call the Python script
% system(sprintf('python "%s"', pythonScript));
system(sprintf('python3 "%s" "%s"', pythonScript, log_file));
