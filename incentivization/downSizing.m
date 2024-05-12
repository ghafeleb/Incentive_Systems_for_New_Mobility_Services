function downSizing(VOT, budget, nonuser_perc_prob)

% VOT = 2.63; % ?????????????????
% budget =  10000;% ?????????????????
% nonuser_perc_prob = 95;% ?????????????????

n_companies_ADMM = 1;
initializeSBaseline = true;
LogicalStr = {'F', 'T'};
iterRun = 250;
iterRun = iterRun + 1;
n_time = 36;
n_time_inc_start_ADMM = 13;
n_time_inc_end_ADMM = 24;
nonuser_prob = nonuser_perc_prob/100;
nonuser_perc_ADMM = repmat(nonuser_prob, n_time, 1);
setting_output_ADMM = sprintf('%.0f', nonuser_perc_ADMM(1)*100);
seedADMM = 2;
seedData = 2;
rho = 20;
fairness = '0_0_0_100_0';
step_size = 0.01;
region_ = 'region_y3_div';
setting_region = '6_9_AVG5_th1_pad_MultipleTimes';
fileName0 = '_StartHour_6_AVG5_pad_theta1e+00';
folderName0 = 'Mar2May_2018_new_5-22_';
% n_iter_ADMM = 5000;
if nonuser_perc_prob == 90 || nonuser_perc_prob == 95 
    n_iter_ADMM = 2000;
else
    n_iter_ADMM = 6000;
end

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
if ~exist(fileADMM, 'file')
    return 
end
load(fileADMM)

clearvars TempS Tempu TempW TempH TempS_time TempW_time TempH_time D_transpose_D_Cell D_tilda_transpose_D_tilda

downSizedFileAddress = fullfile(folderRun, strcat('result_', num2str(iterRun), '.mat'));
save(downSizedFileAddress);