%%%%%% Get speed and volume based on traffic after incentivization
function getSV_realCost(iterRun, nonuser_perc_prob_ADMM0, budget, n_companies, MaxIter2, n_time, n_time_inc_start_ADMM, n_time_inc_end_ADMM)
VOT = 2.63;
setting_region = '5_22_AVG5_th1_pad_MultipleTimes';
% n_time = 204;
% n_time_inc_start_ADMM = 13;
% n_time_inc_end_ADMM = 24;
fprintf('n_companies: %i\n', n_companies)
step_size = 0.01;
iterRun = iterRun+1;
region_ = 'region_toy';
nonuser_perc_prob_ADMM = nonuser_perc_prob_ADMM0/100;
prob_fairness = [0/100, 0/100, 0/100, 100/100, 0/100];
seedADMM = 2;
seedData = 2;
rho = 20;
fprintf('rho: %i\n', rho)
% MaxIter2 = 2000;
LogicalStr = {'F', 'T'};
perc_companies = [0.5; 0.5]; % Percentage of drivers in each company
fprintf('perc_companies: [')
fprintf('%.2f  ', perc_companies)
fprintf(']\n')
setting_perc_companies = strcat(sprintf('%.0f', perc_companies(1)*100), ...
    '_', sprintf('%.0f', perc_companies(2)*100));
initializeSBaseline = true;
fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})

%% Initialization 1
RHSMultiplier = [1, 1.1, 1.5, 2, 2.5];
MaxIter2 = MaxIter2 + seedADMM; % # of iterations of ADMM
% Print norms every normWindow steps
printNorm = false;
fprintf('printNorm: %s\n', LogicalStr{printNorm + 1})
normWindow = 1000;
fprintf('normWindow: %i\n', normWindow)
% Save LP approximations of last iterations
saveLastIters = false;
fprintf('saveLastIters: %s\n', LogicalStr{saveLastIters + 1})
% Percentace of nonuser drivers in each time interval
nonuser_perc_ADMM = repmat(nonuser_perc_prob_ADMM, n_time, 1);
% Initialize the decision matrix S with the baseline
initializeSBaseline = true;
fprintf('initializeSBaseline: %s\n', LogicalStr{initializeSBaseline + 1})
% seedADMM = 2
rng(seedADMM) % Specification of seed for the random number generator (rng)
% Distribution of driver's tt upperbound
% prob_fairness = [0/100, 0/100, 100/100, 0/100, 0/100] % [x1.0, x1.1, x1.5, x2.0, x2.5]
fprintf('RHSMultiplier: [')
fprintf('%.2f  ', RHSMultiplier)
fprintf(']\n')
setting_fairness_output = strcat(sprintf('%.0f', prob_fairness(1)*100), ...
    '_', sprintf('%.0f', prob_fairness(2)*100), ...
    '_', sprintf('%.0f', prob_fairness(3)*100), ...
    '_', sprintf('%.0f', prob_fairness(4)*100), ...
    '_', sprintf('%.0f', prob_fairness(5)*100));
assert(sum(prob_fairness)==1) % Sum of probs in dist should be 1
% VOT = 28/60; % Value of Time (Business)
fprintf('VOT: %.6f\n', VOT)
perc_companies = [0.5; 0.5]; % Percentage of drivers in each company
fprintf('perc_companies: [')
fprintf('%.2f  ', perc_companies)
fprintf(']\n')
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

load(fileIncentivized, 'omega_sol*', 'link_loc', 'tt0_array', 'L_array', 'w_array', 'n_link', 'n_time', 'v_tilda_baseline');
if n_companies==1
    v_users = omega_sol1;
elseif n_companies==2
    v_users = omega_sol1 + omega_sol2;
end
%% Compute volume and speed
tt_single = @(x, v, tt0, w) tt0 + 0.15*tt0*((x+v)^4)/(w^4);
v_incentivized = zeros(n_link*n_time, 1);
s_incentivized = zeros(n_link*n_time, 1);

for iter_omega = 1:n_link*n_time
    tt0 = tt0_array(link_loc(mod(iter_omega-1, size(tt0_array, 1)) + 1, 1), 3);
    w = w_array(link_loc(mod(iter_omega-1, size(w_array, 1)) + 1, 1), 3);
    L = L_array(link_loc(mod(iter_omega-1, size(L_array, 1))+1, 1), 2);
    
    % Volume
    v_dynamic = v_users(iter_omega, 1);
    v_fixed = v_tilda_baseline(iter_omega, 1);
    v_incentivized(iter_omega, 1) = v_fixed + v_dynamic;
    
    % tt
    tt_temp = tt_single(v_dynamic, v_fixed, tt0, w);
    % speed
    s_incentivized(iter_omega, 1) = L/(tt_temp/60);
end
%% Save the speed and volume files
% Fix the idx of speed and volume data & save
volume_reshaped = reshape(v_incentivized(:, 1), n_link, n_time);
volume_reshaped_2save = volume_reshaped(link_loc(:, 1), :);
dlmwrite(fullfile(folderRun, 'v_inc.csv'), volume_reshaped_2save, 'precision', 10);

speed_reshaped = reshape(s_incentivized(:, 1), n_link, n_time);
speed_reshaped_2save = speed_reshaped(link_loc(:, 1), :);
dlmwrite(fullfile(folderRun, 's_inc.csv'), speed_reshaped_2save, 'precision', 10);
