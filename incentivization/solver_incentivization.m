function solver_incentivization(solver_name, MIPGap, solve_binarization, ...
    n_time, n_time_inc_start_solver, n_time_inc_end_solver, ...
    seedData, seedSolverMethod, nonuser_perc, budget, VOT, fairness, ...
    iterRun, step_size, ...
    region_, setting_region, fileName0, folderName0)
fprintf('\nRunning "solver_incentivization"...\n')
n_companies_solver = 1;
iterRun = iterRun+1;
% Result data file name
inputFolder0 = fullfile('../data', region_, setting_region);
folderOutput = fullfile(inputFolder0, ...
                        strcat(solver_name, '_new_Det_initAll2_MultT', ...
                                '_b', num2str(budget), ...
                                '_sD', num2str(seedData), ...
                                '_sS', num2str(seedSolverMethod), ...
                                '_VOT', num2str(VOT), ...
                                '_nC', num2str(n_companies_solver), ...
                                '_f', fairness, ...
                                '_percNonU', num2str(nonuser_perc), ...
                                '_nTIS', num2str(n_time_inc_start_solver), ...
                                '_nTIE', num2str(n_time_inc_end_solver), ...
                                '_ss', num2str(step_size), ...
                                '_itN', num2str(iterRun)));
mkdir(folderOutput);
MIPGapStr = num2str(MIPGap);
filenameOutput = fullfile(folderOutput, strcat(solver_name, '_MIPGap', MIPGapStr, '_solver_result.mat'));
fprintf('filenameOutput: %s\n', filenameOutput)
% % Check if the result already exists
% if exist(filenameOutput, 'file') && check_exist
%     fprintf("Solving the problem is already done.\n")
%     pause(5);
%     return 
% end

% Saving the log showing on command window
log_file = fullfile(folderOutput,  strcat(solver_name, '_MIPGap', ...
                                            MIPGapStr, '_solver_log.txt'));
log_file_pdf = fullfile(folderOutput,  strcat(solver_name, '_MIPGap', ...
                                            MIPGapStr, '_solver_log.pdf'));
if exist(log_file_pdf, 'file')
    return
end
diary(log_file)

%% Load data
% Loading data (especifically user related data) from ADMM algorithm result
% data to prevent the effect of the randomness
% Run ADMM with 0 iterations if the file does not exist
% folderADMM = fullfile(inputFolder0, ...
%             strcat('Det_initAll2_MultT', ...
%             '_b', num2str(0), ...
%             '_sD', num2str(seedData), ...
%             '_sA', num2str(seedSolverMethod), ...
%             '_r', num2str(20), ...
%             '_it', num2str(0),...
%             '_VOT', num2str(VOT), ...
%             '_nC', num2str(1), ...
%             '_f', fairness, ...
%             '_initSB_T', ...
%             '_percNonU', num2str(nonuser_perc), ...
%             '_nTIS', num2str(n_time_inc_start_solver), ...
%             '_nTIE', num2str(n_time_inc_end_solver), ...
%             '_ss', num2str(step_size), ...
%             '_itN', num2str(iterRun)));
fileADMM = strcat('result_', num2str(iterRun), '_ADMMDataInit.mat');
fileADMMAddress = fullfile(folderOutput, fileADMM);
if ~exist(fileADMMAddress, 'file')
    if solver_name=="Gurobi"
        n_iter_ADMM = -1;
    elseif solver_name=="Mosek"
        n_iter_ADMM = -2;
    elseif solver_name=="GLPK"
        n_iter_ADMM = -3;
    end
    ADMM(iterRun-1, nonuser_perc, budget, n_iter_ADMM, ...
        n_time, n_time_inc_start_solver, n_time_inc_end_solver, ...
        VOT, seedSolverMethod, seedData, 20, fairness, step_size, ...
        region_, setting_region, fileName0, folderName0)
end
load(fileADMMAddress, 'tt0_array*', 'w_array*', 'n_OD', 'R_matrix', 'D', ...
    'ones_S', 'v_tilda_baseline', 'RHS', 'idxEndCell', 'gamma', ...
    'n_path', 'n_driver_company_time', 'link_loc', 'q_company_time', ...
    'delta_p_t', 'alpha', 'num_path_v', 'n_link', ...
    'v_dynamic_array_NoInc_allDet'); % ????

% % Free flow travel time
% tt0_array = readmatrix(fullfile('../data', 'capacity', region_, strcat(folderName0, 'link_tt_0_minutes_', region_, '.csv')));
% % Capacity of link
% w_array = readmatrix(fullfile('../data', 'capacity', region_, strcat(folderName0, 'link_capacity_', region_, '.csv')));
% link_loc = readmatrix(fullfile('../data', region_, 'link_loc.txt'));
% % Convert start idx from 0 to 1
% link_loc(:, 1) = link_loc(:, 1) + 1;
% link_loc(:, 2) = link_loc(:, 2) + 1;

% setting_output_ADMM = sprintf('%.0f', nonuser_perc_ADMM(1)*100);
% inputFolder0 = fullfile('../data', region_, setting_region);
% folderRun0 = fullfile(inputFolder0, ...
%     strcat('Det_MultT', ...
%             '_sD', num2str(seedData), ...
%             '_ss', num2str(step_size)));
% outputFolder2 = fullfile(folderRun0, strcat('Run_', num2str(iterRun), '_initEta'));
% fileName2 = fullfile(outputFolder2, strcat('data_', num2str(iterRun), '.mat'));
% load(fileName2, 'n_OD', 'R_matrix', 'D');

%% Print setting
user_perc = 100 - nonuser_perc;
fprintf('Percentage of users: %.2f\n', user_perc)
fprintf('Budget: %d\n', budget)
fprintf('Seed Data: %d\n', seedData)

%% Create required arrays and variables
% v_tilda = v{1, 1} + v{2, 1} + v{3, 1} + v{4, 1};
tt0_array_tiled = repmat(tt0_array(link_loc(:, 1), 3), n_time, 1);
w_array_tiled = repmat(w_array(link_loc(:, 1), 3), n_time, 1);
% % m = size(A{1, 1}, 2);
% % n_user = sum(q_user{1, 1});
% % n_OD = size(q_user{1, 1}, 1);
% 
% R_matrix = sparse(R_matrix);
% 
% ones_S = cell(n_companies_ADMM, n_time);
% for iter_company=1:n_companies_ADMM
%     for iter_time=1:n_time
%         ones_S{iter_company, iter_time} = ones(n_driver_company_time{iter_company, iter_time}, 1);
%     end
% end
%% Create functions
% http://cvxr.com/cvx/doc/funcref.html
% F_v = @(S) A{1, 1} * S * ones(n_user, 1) + v_tilda;

% v_tilda_baseline should come from result data of ADMM algorithm because 
% it depends on the percentage of users. However, the v_tilda_baseline in
% DataLoader_Det and initEta_prestep does not depend on the percentage of users
% F_v = @(S, t) R_matrix(:, (n_time_inc_start_ADMM-1+t)*n_path+1:((n_time_inc_start_ADMM+t)*n_path))*S{1, n_time_inc_start_ADMM+t}*ones_S{1, n_time_inc_start_ADMM+t} + ...
%     v_tilda_baseline(1+(t-1)*n_link:(t+1)*n_link);
F_v_user_t = @(R_matrix,  n_time_inc_start_solver, n_path, S, t) ...
                R_matrix(:, (...
                            n_time_inc_start_solver-1+(t-1))*n_path+1:...
                            ((n_time_inc_start_solver+(t-1))*n_path) ...
                             ) * ...
                                S * ones_S{1, n_time_inc_start_solver+(t-1)};
F_v = @(R_matrix,  n_time_inc_start_solver, n_path, ...
        S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12) ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S1, 1) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S2, 2) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S3, 3) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S4, 4) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S5, 5) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S6, 6) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S7, 7) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S8, 8) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S9, 9) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S10, 10) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S11, 11) + ...
        F_v_user_t(R_matrix,  n_time_inc_start_solver, n_path, S12, 12) + ...
        v_tilda_baseline;
% F_gamma is v*(tt0*(1 + 0.15*(v/w)^4))
% F_gamma = @(S)  0.15*tt0_array_tiled.*((F_v(S)).^5)./(w_array_tiled_pow4) + (F_v(S)).*tt0_array_tiled;

F_gamma_new = @(tt0_array_tiled, w_array_tiled, ...
                R_matrix,  n_time_inc_start_solver, n_path, ...
                S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12) ...
                tt0_array_tiled.*(0.15*...
                    pow_p((F_v(R_matrix,  n_time_inc_start_solver, n_path, ...
                            S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12))./ ...
                                w_array_tiled, 5).*w_array_tiled + ...
                F_v(R_matrix,  n_time_inc_start_solver, n_path, ...
                        S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12));

%% Baseline 
fprintf('Baseline\n')
% [S_baseline, tt_obj_total_baseline] = solve_baseline(user_perc, algo_output_dir, seedData, region_);

% % Check if we need to solve the problem.
% if (budget==0) || (user_perc==0)
%     if (budget==0) && (user_perc==0)
%         fprintf("Both budget and user percentage are zero. Hence, we do not need to solve the problem.\n")
%     elseif (budget==0)
%         fprintf("The budget is zero. Hence, we do not need to solve the problem.\n")
%     else
%         fprintf("The user percentage is zero. Hence, we do not need to solve the problem.\n")
%     end
%     S_sorted = S_baseline;
%     gamma_sol = A{1, 1} * S_sorted * ones(size(S_sorted, 2), 1);
%     % Save variables and overwrite the previous saved file
%     filenameOutput = fullfile(algo_output_dir, strcat(solver_name, '_save.mat'));
%     save(filenameOutput)
%     sound(sin(1:3000));
%     pause(5);
%     return 
% end

%% Running solver
fprintf('Running solver %s\n', solver_name)
rng(seedSolverMethod) % Fix the randomness 
tStartSolver = tic;
if solver_name=="Gurobi"
    % cvx_solver Gurobi
    % https://stackoverflow.com/questions/65324103/matlab-gurobi-in-cvx-solver-fails
    cvx_solver Gurobi_2
    % http://web.cvxr.com/cvx/doc/solver.html#advanced-solver-settings
    cvx_solver_settings('MIPGap',MIPGap);
elseif solver_name=="Mosek"
    cvx_solver Mosek
    % http://web.cvxr.com/cvx/doc/solver.html#advanced-solver-settings
    % https://docs.mosek.com/9.2/toolbox/mip-optimizer.html#termination-criterion
    cvx_solver_settings('MSK_DPAR_MIO_TOL_REL_GAP',MIPGap);
    % cvx_solver_settings('mioObjRelGap',.01);
elseif solver_name=='GLPK'
    cvx_solver GLPK
end
% cvx_begin
% if solve_binarization
%     variable S_sol(m, n_user) binary;
% else
%     variable S_sol(m, n_user);
% end
%     minimize(sum(F_gamma_new(S_sol)))
% %     minimize(0)
%     subject to
%         S_sol' * ones(m, 1) == ones(n_user, 1);
%         D*S_sol*ones(n_user, 1) == q_user{1, 1};
%         c'*S_sol*ones(n_user, 1) <= budget;
%         if solve_binarization~=1
%             S_sol >= 0;
%             S_sol <= 1;
%         end
% cvx_end

cvx_begin
if solve_binarization
    variable S11_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+0}) binary;
    variable S12_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+1}) binary;
    variable S13_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+2}) binary;
    variable S14_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+3}) binary;
    variable S15_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+4}) binary;
    variable S16_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+5}) binary;
    variable S17_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+6}) binary;
    variable S18_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+7}) binary;
    variable S19_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+8}) binary;
    variable S110_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+9}) binary;
    variable S111_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+10}) binary;
    variable S112_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+11}) binary;
else
    variable S11_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+0});
    variable S12_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+1});
    variable S13_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+2});
    variable S14_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+3});
    variable S15_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+4});
    variable S16_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+5});
    variable S17_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+6});
    variable S18_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+7});
    variable S19_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+8});
    variable S110_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+9});
    variable S111_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+10});
    variable S112_binary(n_path, n_driver_company_time{1, n_time_inc_start_solver+11});
end
variable c(1, 1);
minimize(sum(F_gamma_new(tt0_array_tiled, w_array_tiled, ...
                         R_matrix,  n_time_inc_start_solver, n_path, ...
                         S11_binary, S12_binary, S13_binary, ...
                         S14_binary, S15_binary, S16_binary, ...
                         S17_binary, S18_binary, S19_binary, ...
                         S110_binary, S111_binary, S112_binary) + 0));
subject to
if solve_binarization~=1
    S11_binary >= 0;
    S11_binary <= 1;
    S12_binary >= 0;
    S12_binary <= 1;
    S13_binary >= 0;
    S13_binary <= 1;
    S14_binary >= 0;
    S14_binary <= 1;
    S15_binary >= 0;
    S15_binary <= 1;
    S16_binary >= 0;
    S16_binary <= 1;
    S17_binary >= 0;
    S17_binary <= 1;
    S18_binary >= 0;
    S18_binary <= 1;
    S19_binary >= 0;
    S19_binary <= 1;
    S110_binary >= 0;
    S110_binary <= 1;
    S111_binary >= 0;
    S111_binary <= 1;
    S112_binary >= 0;
    S112_binary <= 1;
end
D((n_time_inc_start_solver-1+0)*n_OD+1:n_OD*(n_time_inc_start_solver+0), ((n_time_inc_start_solver-1)+0)*n_path+1:(n_time_inc_start_solver+0)*n_path)*S11_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+0}, 1) == q_company_time{1, n_time_inc_start_solver+0};
D((n_time_inc_start_solver-1+1)*n_OD+1:n_OD*(n_time_inc_start_solver+1), ((n_time_inc_start_solver-1)+1)*n_path+1:(n_time_inc_start_solver+1)*n_path)*S12_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+1}, 1) == q_company_time{1, n_time_inc_start_solver+1};
D((n_time_inc_start_solver-1+2)*n_OD+1:n_OD*(n_time_inc_start_solver+2), ((n_time_inc_start_solver-1)+2)*n_path+1:(n_time_inc_start_solver+2)*n_path)*S13_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+2}, 1) == q_company_time{1, n_time_inc_start_solver+2};
D((n_time_inc_start_solver-1+3)*n_OD+1:n_OD*(n_time_inc_start_solver+3), ((n_time_inc_start_solver-1)+3)*n_path+1:(n_time_inc_start_solver+3)*n_path)*S14_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+3}, 1) == q_company_time{1, n_time_inc_start_solver+3};
D((n_time_inc_start_solver-1+4)*n_OD+1:n_OD*(n_time_inc_start_solver+4), ((n_time_inc_start_solver-1)+4)*n_path+1:(n_time_inc_start_solver+4)*n_path)*S15_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+4}, 1) == q_company_time{1, n_time_inc_start_solver+4};
D((n_time_inc_start_solver-1+5)*n_OD+1:n_OD*(n_time_inc_start_solver+5), ((n_time_inc_start_solver-1)+5)*n_path+1:(n_time_inc_start_solver+5)*n_path)*S16_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+5}, 1) == q_company_time{1, n_time_inc_start_solver+5};
D((n_time_inc_start_solver-1+6)*n_OD+1:n_OD*(n_time_inc_start_solver+6), ((n_time_inc_start_solver-1)+6)*n_path+1:(n_time_inc_start_solver+6)*n_path)*S17_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+6}, 1) == q_company_time{1, n_time_inc_start_solver+6};
D((n_time_inc_start_solver-1+7)*n_OD+1:n_OD*(n_time_inc_start_solver+7), ((n_time_inc_start_solver-1)+7)*n_path+1:(n_time_inc_start_solver+7)*n_path)*S18_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+7}, 1) == q_company_time{1, n_time_inc_start_solver+7};
D((n_time_inc_start_solver-1+8)*n_OD+1:n_OD*(n_time_inc_start_solver+8), ((n_time_inc_start_solver-1)+8)*n_path+1:(n_time_inc_start_solver+8)*n_path)*S19_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+8}, 1) == q_company_time{1, n_time_inc_start_solver+8};
D((n_time_inc_start_solver-1+9)*n_OD+1:n_OD*(n_time_inc_start_solver+9), ((n_time_inc_start_solver-1)+9)*n_path+1:(n_time_inc_start_solver+9)*n_path)*S110_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+9}, 1) == q_company_time{1, n_time_inc_start_solver+9};
D((n_time_inc_start_solver-1+10)*n_OD+1:n_OD*(n_time_inc_start_solver+10), ((n_time_inc_start_solver-1)+10)*n_path+1:(n_time_inc_start_solver+10)*n_path)*S111_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+10}, 1) == q_company_time{1, n_time_inc_start_solver+10};
D((n_time_inc_start_solver-1+11)*n_OD+1:n_OD*(n_time_inc_start_solver+11), ((n_time_inc_start_solver-1)+11)*n_path+1:(n_time_inc_start_solver+11)*n_path)*S112_binary*ones(n_driver_company_time{1, n_time_inc_start_solver+11}, 1) == q_company_time{1, n_time_inc_start_solver+11};
S11_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+0}, 1);
S12_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+1}, 1);
S13_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+2}, 1);
S14_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+3}, 1);
S15_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+4}, 1);
S16_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+5}, 1);
S17_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+6}, 1);
S18_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+7}, 1);
S19_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+8}, 1);
S110_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+9}, 1);
S111_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+10}, 1);
S112_binary'*ones(n_path, 1) == ones(n_driver_company_time{1, n_time_inc_start_solver+11}, 1);
S11_binary'*delta_p_t{n_time_inc_start_solver+0} <= RHS{1}(1:idxEndCell{1, n_time_inc_start_solver});
S12_binary'*delta_p_t{n_time_inc_start_solver+1} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+1}+1:idxEndCell{1, n_time_inc_start_solver+1});
S13_binary'*delta_p_t{n_time_inc_start_solver+2} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+2}+1:idxEndCell{1, n_time_inc_start_solver+2});
S14_binary'*delta_p_t{n_time_inc_start_solver+3} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+3}+1:idxEndCell{1, n_time_inc_start_solver+3});
S15_binary'*delta_p_t{n_time_inc_start_solver+4} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+4}+1:idxEndCell{1, n_time_inc_start_solver+4});
S16_binary'*delta_p_t{n_time_inc_start_solver+5} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+5}+1:idxEndCell{1, n_time_inc_start_solver+5});
S17_binary'*delta_p_t{n_time_inc_start_solver+6} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+6}+1:idxEndCell{1, n_time_inc_start_solver+6});
S18_binary'*delta_p_t{n_time_inc_start_solver+7} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+7}+1:idxEndCell{1, n_time_inc_start_solver+7});
S19_binary'*delta_p_t{n_time_inc_start_solver+8} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+8}+1:idxEndCell{1, n_time_inc_start_solver+8});
S110_binary'*delta_p_t{n_time_inc_start_solver+9} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+9}+1:idxEndCell{1, n_time_inc_start_solver+9});
S111_binary'*delta_p_t{n_time_inc_start_solver+10} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+10}+1:idxEndCell{1, n_time_inc_start_solver+10});
S112_binary'*delta_p_t{n_time_inc_start_solver+11} <= RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+11}+1:idxEndCell{1, n_time_inc_start_solver+11});
c(1) >= alpha(1).*(delta_p_t{n_time_inc_start_solver+0}' * S11_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+0}, 1) +...
                   delta_p_t{n_time_inc_start_solver+1}' * S12_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+1}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+2}' * S13_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+2}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+3}' * S14_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+3}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+4}' * S15_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+4}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+5}' * S16_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+5}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+6}' * S17_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+6}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+7}' * S18_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+7}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+8}' * S19_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+8}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+9}' * S110_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+9}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+10}' * S111_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+10}, 1) + ...
                   delta_p_t{n_time_inc_start_solver+11}' * S112_binary * ones(n_driver_company_time{1, n_time_inc_start_solver+11}, 1) + ...
                   - gamma(1));
c(1) >= 0;
c(1) + 0 <= budget;
%         I have a big S matrix and it contains all the times and all drivers
%         In this step, I design the blocks of S_t for different times by assigning
%         0 to values outside S blocks
% idxEndRow1 = 0;
% idxStartCol = cell(n_companies_solver, n_time);
% idxStartCol(:) = {1};
% idxEndCol = cell(n_companies_solver, n_time);
% idxEndCol(:) = {0};
% nDriverTemp = cell(n_companies_solver, n_time);
% for iter_OD=1:n_OD
%     nPathTemp = num_path_v(iter_OD, 1);
%     for iter_company=1:n_companies_solver
%         for iter_time=1:n_time
%             nDriverTemp{iter_company, iter_time}  = q_company_time{iter_company, iter_time}(iter_OD, 1);
%         end
%     end
%     
%     if nDriverTemp{1, n_time_inc_start_solver+0}>0
%         idxEndCol{1, n_time_inc_start_solver+0} = idxEndCol{1, n_time_inc_start_solver+0} + nDriverTemp{1, n_time_inc_start_solver+0};
%         if iter_OD==1
%             S11_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+0}) == 0;
%         elseif iter_OD == n_OD
%             S11_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+0}:end) == 0;
%         else
%             S11_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+0}:idxEndCol{1, n_time_inc_start_solver+0}) == 0;
%             S11_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+0}:idxEndCol{1, n_time_inc_start_solver+0}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+0} = idxEndCol{1, n_time_inc_start_solver+0} + 1;
%     end
%     
%     if nDriverTemp{1, n_time_inc_start_solver+1}>0
%         idxEndCol{1, n_time_inc_start_solver+1} = idxEndCol{1, n_time_inc_start_solver+1} + nDriverTemp{1, n_time_inc_start_solver+1};
%         if iter_OD==1
%             S12_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+1}) == 0;
%         elseif iter_OD == n_OD
%             S12_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+1}:end) == 0;
%         else
%             S12_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+1}:idxEndCol{1, n_time_inc_start_solver+1}) == 0;
%             S12_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+1}:idxEndCol{1, n_time_inc_start_solver+1}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+1} = idxEndCol{1, n_time_inc_start_solver+1} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+2}>0
%         idxEndCol{1, n_time_inc_start_solver+2} = idxEndCol{1, n_time_inc_start_solver+2} + nDriverTemp{1, n_time_inc_start_solver+2};
%         if iter_OD==1
%             S13_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+2}) == 0;
%         elseif iter_OD == n_OD
%             S13_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+2}:end) == 0;
%         else
%             S13_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+2}:idxEndCol{1, n_time_inc_start_solver+2}) == 0;
%             S13_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+2}:idxEndCol{1, n_time_inc_start_solver+2}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+2} = idxEndCol{1, n_time_inc_start_solver+2} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+3}>0
%         idxEndCol{1, n_time_inc_start_solver+3} = idxEndCol{1, n_time_inc_start_solver+3} + nDriverTemp{1, n_time_inc_start_solver+3};
%         if iter_OD==1
%             S14_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+3}) == 0;
%         elseif iter_OD == n_OD
%             S14_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+3}:end) == 0;
%         else
%             S14_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+3}:idxEndCol{1, n_time_inc_start_solver+3}) == 0;
%             S14_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+3}:idxEndCol{1, n_time_inc_start_solver+3}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+3} = idxEndCol{1, n_time_inc_start_solver+3} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+4}>0
%         idxEndCol{1, n_time_inc_start_solver+4} = idxEndCol{1, n_time_inc_start_solver+4} + nDriverTemp{1, n_time_inc_start_solver+4};
%         if iter_OD==1
%             S15_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+4}) == 0;
%         elseif iter_OD == n_OD
%             S15_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+4}:end) == 0;
%         else
%             S15_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+4}:idxEndCol{1, n_time_inc_start_solver+4}) == 0;
%             S15_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+4}:idxEndCol{1, n_time_inc_start_solver+4}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+4} = idxEndCol{1, n_time_inc_start_solver+4} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+5}>0
%         idxEndCol{1, n_time_inc_start_solver+5} = idxEndCol{1, n_time_inc_start_solver+5} + nDriverTemp{1, n_time_inc_start_solver+5};
%         if iter_OD==1
%             S16_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+5}) == 0;
%         elseif iter_OD == n_OD
%             S16_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+5}:end) == 0;
%         else
%             S16_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+5}:idxEndCol{1, n_time_inc_start_solver+5}) == 0;
%             S16_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+5}:idxEndCol{1, n_time_inc_start_solver+5}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+5} = idxEndCol{1, n_time_inc_start_solver+5} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+6}>0
%         idxEndCol{1, n_time_inc_start_solver+6} = idxEndCol{1, n_time_inc_start_solver+6} + nDriverTemp{1, n_time_inc_start_solver+6};
%         if iter_OD==1
%             S17_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+6}) == 0;
%         elseif iter_OD == n_OD
%             S17_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+6}:end) == 0;
%         else
%             S17_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+6}:idxEndCol{1, n_time_inc_start_solver+6}) == 0;
%             S17_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+6}:idxEndCol{1, n_time_inc_start_solver+6}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+6} = idxEndCol{1, n_time_inc_start_solver+6} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+7}>0
%         idxEndCol{1, n_time_inc_start_solver+7} = idxEndCol{1, n_time_inc_start_solver+7} + nDriverTemp{1, n_time_inc_start_solver+7};
%         if iter_OD==1
%             S18_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+7}) == 0;
%         elseif iter_OD == n_OD
%             S18_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+7}:end) == 0;
%         else
%             S18_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+7}:idxEndCol{1, n_time_inc_start_solver+7}) == 0;
%             S18_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+7}:idxEndCol{1, n_time_inc_start_solver+7}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+7} = idxEndCol{1, n_time_inc_start_solver+7} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+8}>0
%         idxEndCol{1, n_time_inc_start_solver+8} = idxEndCol{1, n_time_inc_start_solver+8} + nDriverTemp{1, n_time_inc_start_solver+8};
%         if iter_OD==1
%             S19_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+8}) == 0;
%         elseif iter_OD == n_OD
%             S19_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+8}:end) == 0;
%         else
%             S19_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+8}:idxEndCol{1, n_time_inc_start_solver+8}) == 0;
%             S19_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+8}:idxEndCol{1, n_time_inc_start_solver+8}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+8} = idxEndCol{1, n_time_inc_start_solver+8} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+9}>0
%         idxEndCol{1, n_time_inc_start_solver+9} = idxEndCol{1, n_time_inc_start_solver+9} + nDriverTemp{1, n_time_inc_start_solver+9};
%         if iter_OD==1
%             S110_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+9}) == 0;
%         elseif iter_OD == n_OD
%             S110_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+9}:end) == 0;
%         else
%             S110_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+9}:idxEndCol{1, n_time_inc_start_solver+9}) == 0;
%             S110_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+9}:idxEndCol{1, n_time_inc_start_solver+9}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+9} = idxEndCol{1, n_time_inc_start_solver+9} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+10}>0
%         idxEndCol{1, n_time_inc_start_solver+10} = idxEndCol{1, n_time_inc_start_solver+10} + nDriverTemp{1, n_time_inc_start_solver+10};
%         if iter_OD==1
%             S111_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+10}) == 0;
%         elseif iter_OD == n_OD
%             S111_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+10}:end) == 0;
%         else
%             S111_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+10}:idxEndCol{1, n_time_inc_start_solver+10}) == 0;
%             S111_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+10}:idxEndCol{1, n_time_inc_start_solver+10}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+10} = idxEndCol{1, n_time_inc_start_solver+10} + 1;
%     end
%     if nDriverTemp{1, n_time_inc_start_solver+11}>0
%         idxEndCol{1, n_time_inc_start_solver+11} = idxEndCol{1, n_time_inc_start_solver+11} + nDriverTemp{1, n_time_inc_start_solver+11};
%         if iter_OD==1
%             S112_binary(idxEndRow1+nPathTemp+1:end, 1:nDriverTemp{1, n_time_inc_start_solver+11}) == 0;
%         elseif iter_OD == n_OD
%             S112_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+11}:end) == 0;
%         else
%             S112_binary(1:idxEndRow1, idxStartCol{1, n_time_inc_start_solver+11}:idxEndCol{1, n_time_inc_start_solver+11}) == 0;
%             S112_binary(idxEndRow1+nPathTemp+1:end, idxStartCol{1, n_time_inc_start_solver+11}:idxEndCol{1, n_time_inc_start_solver+11}) == 0;
%         end
%         idxStartCol{1, n_time_inc_start_solver+11} = idxEndCol{1, n_time_inc_start_solver+11} + 1;
%     end
%     idxEndRow1 = idxEndRow1 + nPathTemp;
% end
cvx_end

% time_run = toc(tStartSolver);

S_binary_full = cell(n_companies_solver, n_time);
S_binary_full{1, n_time_inc_start_solver+0} = full(S11_binary);
S_binary_full{1, n_time_inc_start_solver+1} = full(S12_binary);
S_binary_full{1, n_time_inc_start_solver+2} = full(S13_binary);
S_binary_full{1, n_time_inc_start_solver+3} = full(S14_binary);
S_binary_full{1, n_time_inc_start_solver+4} = full(S15_binary);
S_binary_full{1, n_time_inc_start_solver+5} = full(S16_binary);
S_binary_full{1, n_time_inc_start_solver+6} = full(S17_binary);
S_binary_full{1, n_time_inc_start_solver+7} = full(S18_binary);
S_binary_full{1, n_time_inc_start_solver+8} = full(S19_binary);
S_binary_full{1, n_time_inc_start_solver+9} = full(S110_binary);
S_binary_full{1, n_time_inc_start_solver+10} = full(S111_binary);
S_binary_full{1, n_time_inc_start_solver+11} = full(S112_binary);

time_run = toc(tStartSolver);

% S_sol_full = full(S_sol);

S_check_response = cell(n_companies_solver, n_time);
for time_idx=0:11
    if solve_binarization
        % All the values are 0 or 1 but some 0 and 1 values have different
        % decimals in high precision. Something like 1.00000001. To prevent
        % issues, we round all the numbers
        S_sol_full = round(S_binary_full{1, n_time_inc_start_solver+time_idx});
%         [row_S_sorted, col_S_sorted] = find(S_sol_full==1);
%         [out_temp, idx_S_sorted] = sort(row_S_sorted);
%         S_sorted = S_sol_full(:, idx_S_sorted);
%         S_check_response{1, n_time_inc_start_solver + time_idx} = S_sorted;
        S_check_response{1, n_time_inc_start_solver + time_idx} = transpose(sortrows(transpose(round(S_sol_full)),'descend'));
    else
        S_sol_full = round(S_binary_full{1, n_time_inc_start_solver+time_idx});
        S_check_response{1, n_time_inc_start_solver + time_idx} = S_sol_full;
    end
end

%% Compute total travel time after binarization
tt = @(x, v, tt0, w) (x+v)*tt0 + 0.15*tt0*((x+v)^5)/(w^4);
omega_sol1 =  ...
    R_matrix(:, (n_time_inc_start_solver-1+0)*n_path+1:((n_time_inc_start_solver+0)*n_path))*S_check_response{1, n_time_inc_start_solver+0}*ones_S{1, n_time_inc_start_solver+0} +...
    R_matrix(:, (n_time_inc_start_solver-1+1)*n_path+1:((n_time_inc_start_solver+1)*n_path))*S_check_response{1, n_time_inc_start_solver+1}*ones_S{1, n_time_inc_start_solver+1} +...
    R_matrix(:, (n_time_inc_start_solver-1+2)*n_path+1:((n_time_inc_start_solver+2)*n_path))*S_check_response{1, n_time_inc_start_solver+2}*ones_S{1, n_time_inc_start_solver+2} +...
    R_matrix(:, (n_time_inc_start_solver-1+3)*n_path+1:((n_time_inc_start_solver+3)*n_path))*S_check_response{1, n_time_inc_start_solver+3}*ones_S{1, n_time_inc_start_solver+3} +...
    R_matrix(:, (n_time_inc_start_solver-1+4)*n_path+1:((n_time_inc_start_solver+4)*n_path))*S_check_response{1, n_time_inc_start_solver+4}*ones_S{1, n_time_inc_start_solver+4} +...
    R_matrix(:, (n_time_inc_start_solver-1+5)*n_path+1:((n_time_inc_start_solver+5)*n_path))*S_check_response{1, n_time_inc_start_solver+5}*ones_S{1, n_time_inc_start_solver+5} +...
    R_matrix(:, (n_time_inc_start_solver-1+6)*n_path+1:((n_time_inc_start_solver+6)*n_path))*S_check_response{1, n_time_inc_start_solver+6}*ones_S{1, n_time_inc_start_solver+6} +...
    R_matrix(:, (n_time_inc_start_solver-1+7)*n_path+1:((n_time_inc_start_solver+7)*n_path))*S_check_response{1, n_time_inc_start_solver+7}*ones_S{1, n_time_inc_start_solver+7} +...
    R_matrix(:, (n_time_inc_start_solver-1+8)*n_path+1:((n_time_inc_start_solver+8)*n_path))*S_check_response{1, n_time_inc_start_solver+8}*ones_S{1, n_time_inc_start_solver+8} +...
    R_matrix(:, (n_time_inc_start_solver-1+9)*n_path+1:((n_time_inc_start_solver+9)*n_path))*S_check_response{1, n_time_inc_start_solver+9}*ones_S{1, n_time_inc_start_solver+9} +...
    R_matrix(:, (n_time_inc_start_solver-1+10)*n_path+1:((n_time_inc_start_solver+10)*n_path))*S_check_response{1, n_time_inc_start_solver+10}*ones_S{1, n_time_inc_start_solver+10} +...
    R_matrix(:, (n_time_inc_start_solver-1+11)*n_path+1:((n_time_inc_start_solver+11)*n_path))*S_check_response{1, n_time_inc_start_solver+11}*ones_S{1, n_time_inc_start_solver+11} +...
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
tt_obj_total_NoInc_7_8 = sum(tt_obj_NoInc(n_link*n_time_inc_start_solver+1:n_link*n_time_inc_end_solver))/60;
fprintf('Travel time between 7AM and 8AM before incentivization: %.6f hours\n', tt_obj_total_NoInc_7_8)
tt_obj_7_8 = sum(tt_obj(n_link*n_time_inc_start_solver+1:n_link*n_time_inc_end_solver))/60;
fprintf('Travel time between 7AM and 8AM after incentivization: %.6f hours\n', tt_obj_7_8)
tt_obj_total_NoInc_7_830 = sum(tt_obj_NoInc(n_link*12+1:n_link*30))/60;
fprintf('Travel time between 7AM and 8:30AM before incentivization: %.6f hours\n', tt_obj_total_NoInc_7_830)
tt_obj_7_830 = sum(tt_obj(n_link*12+1:n_link*30))/60;
fprintf('Travel time between 7AM and 8:30AM after incentivization: %.6f hours\n', tt_obj_7_830)
tt_obj_user_NoInc = sum(tt_obj_user_NoInc)/60;
fprintf('User total travel time before incentivization: %.6f hours\n', tt_obj_user_NoInc)
tt_obj_user_ADMM = sum(tt_obj_user_ADMM)/60;
fprintf('User total travel time after incentivization: %.6f hours\n', tt_obj_user_ADMM)
%% Check results
const1_gap_positive = cell(n_companies_solver, n_time);
const2_gap_positive = cell(n_companies_solver, n_time);
const3_gap_positive = cell(n_companies_solver, n_time);
for time_idx=0:11
    ones_S1 = ones(size(S_check_response{1, n_time_inc_start_solver + time_idx}, 1), 1);
    ones_S2 = ones(size(S_check_response{1, n_time_inc_start_solver + time_idx}, 2), 1);
    
    % Check the number of violated constraints (group 3)
    const1_gap = D((n_time_inc_start_solver + time_idx)*n_OD+1:n_OD*(n_time_inc_start_solver+time_idx+1), ((n_time_inc_start_solver-1)+time_idx)*n_path+1:(n_time_inc_start_solver+time_idx)*n_path) * ...
                    S_check_response{1, n_time_inc_start_solver + time_idx} * ...
                    ones_S2 - ...
                    q_company_time{1, n_time_inc_start_solver + time_idx + 1};
    const1_gap_positive{1, n_time_inc_start_solver + time_idx} = const1_gap(const1_gap~=0);
    
    % Check the number of violated constraints (group 1)
    const2_gap = S_check_response{1, n_time_inc_start_solver + time_idx}' * ...
                    ones_S1 - ones_S2;
    const2_gap_positive{1, n_time_inc_start_solver + time_idx} =  const2_gap(const2_gap~=0);        

    if time_idx == 0
        RHS_iter = RHS{1}(1:idxEndCell{1, n_time_inc_start_solver});
    else
        RHS_iter = RHS{1}(idxEndCell{1, (n_time_inc_start_solver-1)+time_idx}+1:idxEndCell{1, n_time_inc_start_solver+time_idx});
    end
    const3_gap = S_check_response{1, n_time_inc_start_solver + time_idx}' * ...
                    delta_p_t{n_time_inc_start_solver+time_idx} - RHS_iter;
    const3_gap_positive{1, n_time_inc_start_solver + time_idx} = const3_gap(const3_gap~=0);
    
    % Check the total travel time
%     tt = @(x, v, tt0, w) (x+v)*tt0 + 0.15*tt0*((x+v)^5)/(w^4);
%     ones_S2 = ones(size(S_check_response, 2), 1);
%     gamma_sol = A{1, 1} * S_check_response * ones_S2;
%     tt_obj = zeros(size(A{1, 1}, 1), 1);
%     for iter_gamma = 1:size(gamma_sol, 1)
%         tt0 = tt0_array(link_loc(mod(iter_gamma-1, size(tt0_array, 1)) + 1, 1), 3);
%         w = w_array(link_loc(mod(iter_gamma-1, size(w_array, 1)) + 1, 1), 3);
%         v_iter = v{1, 1}(iter_gamma) + v{2, 1}(iter_gamma) + v{3, 1}(iter_gamma) + v{4, 1}(iter_gamma); 
%         tt_obj(iter_gamma) = tt(gamma_sol(iter_gamma, 1), v_iter, tt0, w);
%     end
%     tt_obj_total = sum(tt_obj)/60;
end

% Check the total travel time
tt_obj2 = F_gamma_new(tt0_array_tiled, w_array_tiled, ...
                                 R_matrix,  n_time_inc_start_solver, n_path, ...
                                 S_check_response{1, n_time_inc_start_solver + 0}, ...
                                S_check_response{1, n_time_inc_start_solver + 1}, ...
                                S_check_response{1, n_time_inc_start_solver + 2}, ...
                                S_check_response{1, n_time_inc_start_solver + 3}, ...
                                S_check_response{1, n_time_inc_start_solver + 4}, ...
                                S_check_response{1, n_time_inc_start_solver + 5}, ...
                                S_check_response{1, n_time_inc_start_solver + 6}, ...
                                S_check_response{1, n_time_inc_start_solver + 7}, ...
                                S_check_response{1, n_time_inc_start_solver + 8}, ...
                                S_check_response{1, n_time_inc_start_solver + 9}, ...
                                S_check_response{1, n_time_inc_start_solver + 10}, ...
                                S_check_response{1, n_time_inc_start_solver + 11}) + 0;
tt_obj_total2 = sum(tt_obj2)/60;

RHS_cost_constraint = 0;
for time_idx=0:11
    RHS_cost_constraint = RHS_cost_constraint + ...
                                delta_p_t{n_time_inc_start_solver + time_idx}' * ...
                                S_check_response{1, n_time_inc_start_solver + time_idx} * ...  
                                ones(n_driver_company_time{1, n_time_inc_start_solver + time_idx}, 1);
end
cost_lower_gap = c(1) - alpha(1)*(RHS_cost_constraint - gamma(1));

cost_budget_gap = budget - c(1);

% Violation of zero elements
idxEndRow1 = 0;
idxStartCol = cell(n_companies_solver, n_time);
idxStartCol(:) = {1};
idxEndCol = cell(n_companies_solver, n_time);
idxEndCol(:) = {0};
nDriverTemp = cell(n_companies_solver, n_time);
gap_nonzero = cell(n_companies_solver, n_OD);
idxEndCol(:) = {0};
for iter_OD=1:n_OD
    nPathTemp = num_path_v(iter_OD, 1);
    for iter_company=1:n_companies_solver
        for iter_time=1:n_time
            nDriverTemp{iter_company, iter_time} = q_company_time{iter_company, iter_time}(iter_OD, 1);
        end
    end
    for iter_time=0:11
        if nDriverTemp{1, n_time_inc_start_solver+iter_time}>0
            idxEndCol{1, n_time_inc_start_solver+iter_time} = ...
                        idxEndCol{1, n_time_inc_start_solver+iter_time} + ...
                        nDriverTemp{1, n_time_inc_start_solver+iter_time};
            if iter_OD==1
                gap_nonzero{iter_company, iter_OD} = gap_nonzero{iter_company, iter_OD} + ...
                            sum(sum(S_check_response{1, n_time_inc_start_solver+iter_time}(idxEndRow1+nPathTemp+1:end, ...
                                    1:nDriverTemp{1, n_time_inc_start_solver+iter_time} ~= 0)));
            elseif iter_OD == n_OD
                gap_nonzero{iter_company, iter_OD} = gap_nonzero{iter_company, iter_OD} + ...
                            sum(sum(S_check_response{1, n_time_inc_start_solver+iter_time}(1:idxEndRow1, ...
                                    idxStartCol{1, n_time_inc_start_solver+iter_time}:end) ~= 0));
            else
                gap_nonzero{iter_company, iter_OD} = gap_nonzero{iter_company, iter_OD} + ...
                            sum(sum(S_check_response{1, n_time_inc_start_solver+iter_time}(1:idxEndRow1, ...
                                    idxStartCol{1, n_time_inc_start_solver+iter_time}:idxEndCol{1, n_time_inc_start_solver+iter_time}) ~= 0));
                gap_nonzero{iter_company, iter_OD} = gap_nonzero{iter_company, iter_OD} + ...
                            sum(sum(S_check_response{1, n_time_inc_start_solver+iter_time}(idxEndRow1+nPathTemp+1:end, ...
                                    idxStartCol{1, n_time_inc_start_solver+iter_time}:idxEndCol{1, n_time_inc_start_solver+iter_time}) ~= 0));
            end
            idxStartCol{1, n_time_inc_start_solver+iter_time} = idxEndCol{1, n_time_inc_start_solver+iter_time} + 1;
        end
    end
end

%% Gaps of constraints
load(fileADMMAddress, 'n_driver_company', 'B', 'q_company', 'etaDet', ...
    'delta_pDet', 'q', 'q_nonuser', 'q_user'); % ????
S_binary_full_blkdiag = cell(n_companies_solver, 1);
% S_binary_full_blkdiag{1, 1} = blkdiag(S_check_response{1, :});
S_binary_full_blkdiag{1, 1} = round(blkdiag(S_check_response{1, :}));
S_binary_full_blkdiag{1, 1} = transpose(sortrows(transpose(S_binary_full_blkdiag{1, 1}),'descend'));

const11_gap =  [zeros((n_time_inc_start_solver-1)*n_OD, 1);...
    D((n_time_inc_start_solver-1)*n_OD+1:n_time_inc_end_solver*n_OD, (n_time_inc_start_solver-1)*n_path+1:n_time_inc_end_solver*n_path)*S_binary_full_blkdiag{1, 1}*...
    ones(n_driver_company{1}, 1); zeros((n_time-n_time_inc_end_solver)*n_OD, 1)] - q_company{1};
sum_const11_gap = sum(abs(const11_gap));
fprintf('Gap of number of drivers of ODs constraint 1: %.6f\n', sum_const11_gap)
fprintf("\n")
const21_gap = S_binary_full_blkdiag{1, 1}'*ones(n_path*(n_time_inc_end_solver-n_time_inc_start_solver+1), 1) - ones(n_driver_company{1}, 1);
sum_const21_gap = sum(abs(const21_gap));
fprintf('Gap of single path assignment constraint 1: %.6f\n', sum_const21_gap)
fprintf("\n")


% RHS of baseline
RHS_baseline = cell(n_companies_solver, 1);
for iter_company=1:n_companies_solver
    RHS_baseline{iter_company} = B{iter_company} * etaDet;
end
nDeviatedCompany = cell(n_companies_solver, 1);
nDeviatedCompany(:) = {0};
nDeviatedCompanyOD = cell(n_companies_solver, n_OD);
nDeviatedCompanyOD(:) = {0};
deviatedOD = cell(n_companies_solver, 1);
deviatedOD(:) = {[]};
for iter_company=1:n_companies_solver
    idxEndRow1 = 0;
    idxStartCol = 1;
    idxEndCol = 0;
    for iter_time=n_time_inc_start_solver:n_time_inc_end_solver
        for iter_OD=1:n_OD
            nPathTemp = num_path_v(iter_OD, 1);
            nDriverTemp  = q_company_time{iter_company, iter_time}(iter_OD, 1);
            if nDriverTemp>0
                idxEndCol = idxEndCol + nDriverTemp;
                nDeviatedTemp = 0;
                for iter_driver=1:nDriverTemp
                    if round(sum(S_binary_full_blkdiag{iter_company, 1}((idxEndRow1+1):(idxEndRow1+nPathTemp), idxStartCol+(iter_driver-1))), 3) ~= 1
                        fprintf("%g\n", sum(S_binary_full_blkdiag{iter_company, 1}((idxEndRow1+1):(idxEndRow1+nPathTemp), idxStartCol+(iter_driver-1))))
                    end
                    
                    assert(round(sum(S_binary_full_blkdiag{iter_company, 1}((idxEndRow1+1):(idxEndRow1+nPathTemp), idxStartCol+(iter_driver-1))), 3) == 1.0);
                    if ((S_binary_full_blkdiag{iter_company, 1}(:, idxStartCol+(iter_driver-1)))'*delta_pDet(n_path*(n_time_inc_start_solver-1)+1:n_path*n_time_inc_end_solver) - RHS_baseline{iter_company}(idxStartCol+(iter_driver-1), 1))>0
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
for iter_company=1:n_companies_solver
    for iter_OD=1:n_OD
        if nDeviatedCompanyOD{iter_company, iter_OD}>0
            fprintf('# of deviated drivers of company %i in OD %i: %i drivers\n', iter_company, iter_OD, nDeviatedCompanyOD{iter_company, iter_OD})
        end
    end
    fprintf('\n')
end
fprintf('Assertion of order of drivers in assignment is PASSED.\n')

gap_baseline1 = S_binary_full_blkdiag{1, 1}'*delta_pDet(n_path*(n_time_inc_start_solver-1)+1:n_path*n_time_inc_end_solver) - RHS_baseline{1};
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
const31_gap = S_binary_full_blkdiag{1, 1}'*delta_pDet(n_path*(n_time_inc_start_solver-1)+1:n_path*n_time_inc_end_solver) - RHS{1};
sum_abs_const31_gap = sum(abs(const31_gap));
const31_gap_positive = const31_gap(const31_gap>0);
const31_gap_negative = const31_gap(const31_gap<0);
sum_pos_const31_gap = sum((const31_gap_positive));
sum_neg_const31_gap = sum((const31_gap_negative));
fprintf('Absoulte gap of minimum travel time constraint 1: %.6f hours\n', sum_abs_const31_gap/60)
fprintf('Positive deviation from minimum travel time constraint 1: %.6f hours\n', sum_pos_const31_gap/60)
fprintf('Negative deviation from minimum travel time constraint 1: %.6f hours\n', sum_neg_const31_gap/60)
fprintf("\n")
gap_cost1 = c(1) - alpha(1).*(delta_pDet(n_path*(n_time_inc_start_solver-1)+1:n_path*n_time_inc_end_solver)'*S_binary_full_blkdiag{1, 1}*ones(n_driver_company{1}, 1)-gamma(1));
cost_1 = alpha(1).*(delta_pDet(n_path*(n_time_inc_start_solver-1)+1:n_path*n_time_inc_end_solver)'*S_binary_full_blkdiag{1, 1}*ones(n_driver_company{1}, 1)-gamma(1));
fprintf('c(1) (LP cost variable of company 1): %.6f\n', c(1))
fprintf('Cost of company 1 (Result of LP): %.6f\n', cost_1)
fprintf('Gap of cost constraint 1 (LP): %.6f\n', gap_cost1)
fprintf('Total budget: %.4f\n', budget)
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
%% Print results
fprintf('\n\nPercentage of users: %.2f\n', user_perc)
fprintf('Budget: %d\n', budget)
fprintf('Seed Data: %d\n', seedData)
fprintf('Computation time: %.4f minutes\n', time_run/60.0)
for time_idx=0:11
    fprintf('Number of violations of constraint 1: %d\n', ...
                size(const1_gap_positive{1, n_time_inc_start_solver + time_idx}, 1))
    fprintf('Number of violations of constraint 2: %d\n', ...
                size(const2_gap_positive{1, n_time_inc_start_solver + time_idx}, 1))
    fprintf('Number of violations of constraint 2: %d\n', ...
                size(const3_gap_positive{1, n_time_inc_start_solver + time_idx}, 1))
end
fprintf('Total travel time: %.2f hours', tt_obj_total2)
tt_obj_total2 = sum(tt_obj2)/60;
fprintf('Total travel time after incentivization: %.6f hours\n', tt_obj_total2)
tt_obj_7_8 = sum(tt_obj2(n_link*n_time_inc_start_solver+1:n_link*n_time_inc_end_solver))/60;
fprintf('Travel time between 7AM and 8AM after incentivization: %.6f hours\n', tt_obj_7_8)
tt_obj_7_830 = sum(tt_obj2(n_link*12+1:n_link*30))/60;

fprintf('\n*** Travel time between 7AM and 8:30AM after incentivization: %.6f hours ***\n\n', tt_obj_7_830)
fprintf('Computation time: %.4f minutes\n', time_run/60.0)

fprintf('Cost: %.2f \n', c(1))
fprintf('cost_lower_gap: %.2f \n', cost_lower_gap)
fprintf('cost_budget_gap: %.2f \n', cost_budget_gap)
% for idxInc = 1:n_inc
%     incentive_temp = incentives(idxInc);
%     fprintf("%d drivers with %d dollars incentive\n", n_driver_temp(idxInc), incentive_temp)
% end
% fprintf("Average spent incentive per all drivers in all times: %.2f\n", avg1)
% fprintf("Average spent incentive per all drivers in 1st time: %.2f\n", avg2)
% fprintf("Average spent incentive per all user drivers: %.2f\n", avg3)
% fprintf("Average spent incentive per incentivized drivers: %.2f\n", avg4)

%% Keep record of runtime
% Define the file name
fileNameRuntime = fullfile(folderOutput, 'runtimes.csv');
% Check if the file exists
if isfile(fileNameRuntime)
    % File exists, load existing data
    existingRuntime = readmatrix(fileNameRuntime);
    % Append new data
    updatedRuntime = [existingRuntime; time_run/60.0];
else
    % File does not exist, initialize with newData
    updatedRuntime = time_run/60.0;
end

% Save updated data to CSV
writematrix(updatedRuntime, fileNameRuntime);

%% Save Files
fprintf('Save results\n')    
pause(10);
save(filenameOutput)
sound(sin(1:3000));

diary off; % Stop logging
% Python script location
pythonScript = fullfile('../lib', 'convert_to_pdf.py');
% Call the Python script
% system(sprintf('python "%s"', pythonScript));
system(sprintf('python3 "%s" "%s"', pythonScript, log_file));