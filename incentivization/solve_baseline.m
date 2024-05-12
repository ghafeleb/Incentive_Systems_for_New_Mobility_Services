function [S_baseline, tt_obj_total_baseline] = solve_baseline(user_perc, algo_output_dir, seedData, region_)

%% Load Data File
fprintf('Load Data File\n')
pause(5);
percU_folder_name = strcat('percU', num2str(user_perc, "%.2f"));
% filename = fullfile('data', region_, time_inc_setting, percU_folder_name, strcat('AllVar_sD', num2str(seedData), '.mat'));
filename = fullfile(algo_output_dir, strcat('AllVar_sD', num2str(seedData), '.mat'));
fprintf('filename: %s\n', filename)
pause(5);
load(filename, 'A', 'v', 'I', 'm', 'q_user');

tt0_array = readmatrix(fullfile('data', region_, strcat('Mar2May_2018_new_5-22_link_tt_0_minutes_', region_, '.csv')));
w_array = readmatrix(fullfile('data', region_, strcat('Mar2May_2018_new_5-22_link_capacity_', region_, '.csv')));
link_loc = readmatrix(fullfile('data', region_, 'link_loc.txt'));
% Convert starting index from 0 to 1
link_loc(:, 1) = link_loc(:, 1) + 1;
link_loc(:, 2) = link_loc(:, 2) + 1;

m{1, 1} = size(A{1, 1}, 2);
n_user = sum(q_user{1, 1});
n_OD = size(q_user{1, 1}, 1);

%% Solution
fprintf('Solution\n')
pause(2);
% Creating baseline assignment matrix of users
S_baseline = zeros(m{1, 1}, n_user);
counter_driver = 1;
for iter_OD=1:n_OD
    index_min = I{1, 1}(iter_OD); 
    for iter_driver=1:q_user{1, 1}(iter_OD)
        S_baseline(index_min, counter_driver) = 1;
        counter_driver = counter_driver + 1;
    end
end

% Compute travel time of baseline
tt = @(x, v, tt0, w) (x+v)*tt0 + 0.15*tt0*((x+v)^5)/(w^4);
ones_S1 = ones(size(S_baseline, 1), 1);
ones_S2 = ones(size(S_baseline, 2), 1);
gamma_sol_baseline = A{1, 1}*S_baseline*ones_S2;
tt_obj_baseline = zeros(size(A{1, 1}, 1), 1);
v_total_baseline = zeros(size(gamma_sol_baseline, 1), 1);
for iter_gamma = 1:size(gamma_sol_baseline, 1)
    tt0 = tt0_array(link_loc(mod(iter_gamma-1, size(tt0_array, 1)) + 1, 1), 3);
    w = w_array(link_loc(mod(iter_gamma-1, size(w_array, 1)) + 1, 1), 3);
    v_iter = v{1, 1}(iter_gamma) + v{2, 1}(iter_gamma) + v{3, 1}(iter_gamma) + v{4, 1}(iter_gamma); 
    v_total_baseline(iter_gamma, 1) = v_iter + gamma_sol_baseline(iter_gamma, 1);
    tt_obj_baseline(iter_gamma) = tt(gamma_sol_baseline(iter_gamma, 1), v_iter, tt0, w);
end
tt_obj_total_baseline = sum(tt_obj_baseline)/60;
fprintf('\nTotal travel time of BASELINE: %.2f hours\n', tt_obj_total_baseline)

