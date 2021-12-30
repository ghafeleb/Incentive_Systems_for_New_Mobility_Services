
function initEta_prestep(iterRun, nonuser_perc_prob0, n_time, n_time_inc_start, n_time_inc_end)
step_size_start = 0.01;
iterRun = iterRun+1;
seedData = 2;
rng(seedData)
nonuser_perc_prob = nonuser_perc_prob0/100;
% nonuser_perc_prob = 0;
prob_fairness = [0/100, 0/100, 0/100, 100/100, 0/100];
LogicalStr = {'F', 'T'};
perc_companies = [0.5; 0.5]; % Percentage of drivers in each company
setting_perc_companies = strcat(sprintf('%.0f', perc_companies(1)*100), ...
                                '_', sprintf('%.0f', perc_companies(2)*100));

region_ = 'region_toy';
setting_region = '5_22_AVG5_th1_pad_MultipleTimes';
% fileName = '_StartHour_6_AVG5_pad_theta1e+00';
fileName0 = '_StartHour_5_AVG5_pad_theta1e+00';
% n_time = 12*3;
% n_time_inc_start = 13;
% n_time_inc_end = 48;
n_companies = 2;
setting_fairness_output = strcat(sprintf('%.0f', prob_fairness(1)*100), ...
                            '_', sprintf('%.0f', prob_fairness(2)*100), ...
                            '_', sprintf('%.0f', prob_fairness(3)*100), ...
                            '_', sprintf('%.0f', prob_fairness(4)*100), ...
                            '_', sprintf('%.0f', prob_fairness(5)*100));
assert(sum(prob_fairness)==1) % Sum of probs in dist should be 1

%% Data preparation for ADMM
nonuser_perc = repmat(nonuser_perc_prob, n_time, 1);
setting_output = sprintf('%.0f', nonuser_perc(1)*100);
inputFolder0 = fullfile('data', region_, setting_region); 
folderRunDet = fullfile('data', region_, setting_region, ...
                    strcat('Det_MultT', ...
                    '_sD', num2str(seedData), ...
                    '_nC', num2str(n_companies), ...
                    '_f', setting_fairness_output, ...
                    '_percC', setting_perc_companies, ...
                    '_percNonU', setting_output, ...
                    '_ss', num2str(step_size_start)));

inputFolder1 = fullfile(inputFolder0, 'initialData'); 

if ~(iterRun==1)
    inputFolder2Det = fullfile(folderRunDet, strcat('Run_', num2str(iterRun)));
end
D = readmatrix(fullfile(inputFolder1, strcat('D', fileName0, '.txt')));
w_array = readmatrix(fullfile('data', 'capacity', region_, strcat('Mar2May_2018_new_5-22_link_capacity_', region_, '.csv')));
link_loc = readmatrix(fullfile('data', region_, 'link_loc.txt'));
% Convert start idx from 0 to 1
link_loc(:, 1) = link_loc(:, 1) + 1;
link_loc(:, 2) = link_loc(:, 2) + 1;
n_link = size(w_array, 1);



%% Arrays of travel time Deterministic
% # of paths of each OD
num_path_v = readmatrix(fullfile(inputFolder1, 'num_path_v.txt'));
% TT of each path. Size: (max_num_path*n_time)x(n_OD)
if iterRun==1
    tt_pathDet = readmatrix(fullfile(inputFolder1, strcat('tt', fileName0, '.txt')))/60; % Seconds to minutes
else
    tt_pathDet = readmatrix(fullfile(inputFolder2Det, strcat('tt', fileName0, '.txt')))/60; % Seconds to minutes
end
max_num_path = max(num_path_v);
n_path = sum(num_path_v);
n_OD = size(tt_pathDet, 2);
% etaDet = zeros(n_OD*n_time, 1); 
% delta_pDet = zeros(n_path*n_time, 1);
I_minTT = cell(n_time, 1); 
I_minTT(:) = {zeros(n_OD, 1)};
for iter_time=1:n_time
    % Create delta_pDet
    index_0 = tt_pathDet(max_num_path*(iter_time-1)+1, :)==0;
    tt_pathDet(max_num_path*(iter_time-1)+1, index_0) = 0.0001;
    temp_delta = tt_pathDet(max_num_path*(iter_time-1)+1:max_num_path*(iter_time), :);
    temp_delta = temp_delta(:);
    temp_delta(temp_delta==0) = [];
    delta_pDet(n_path*(iter_time-1)+1:n_path*(iter_time)) = temp_delta;
    
    idxTemp = 0; % Starting idx of path
    % Create etaDet
    row_index = max_num_path*(iter_time-1)+1:max_num_path*(iter_time);
    for iter_OD=1:n_OD
        tempMinTT = tt_pathDet(row_index, iter_OD);
        tempMinTT(tempMinTT==0) = [];
        [tempMinTT, tempMinTTIdx] = min(tempMinTT);
        etaDet((iter_time-1)*n_OD+iter_OD, 1) = tempMinTT;
        I_minTT{iter_time, 1}(iter_OD, 1) = idxTemp + tempMinTTIdx;
        idxTemp = idxTemp + num_path_v(iter_OD, 1);
    end
end
delta_pDet(delta_pDet==0.0001) = 0;
etaDet(etaDet==0.0001) = 0;
clearvars index_0 temp_delta row_index tempMinTT iter_time iter_OD delta_pDet etaDet
% clearvars row_index tempMinTT iter_time iter_OD

%% Create baseline decision matrix and volume vector of users and nonusers, Deterministic
R = cell(n_time, 1);
R_temp = cell(n_time, 1);
% A = cell(n_time, 1);
% P = cell(n_time, 1);
for iter_time=1:n_time
    if iterRun==1
        R{iter_time, 1} = readmatrix(fullfile(inputFolder1, strcat('R', num2str(iter_time), fileName0, '.txt')));
    else
        R{iter_time, 1} = readmatrix(fullfile(inputFolder2Det, strcat('R', num2str(iter_time), fileName0, '.txt')));
    end
    R_temp{iter_time, 1} = [zeros(n_link*(iter_time-1), n_path); R{iter_time, 1}];
end
R_matrix = cat(2, R_temp{:});

%% OD
outputFolder2 = fullfile(folderRunDet, strcat('Run_', num2str(1)));
if iterRun==1
    q_time = cell(n_time, 1);
    q_nonuser_time = cell(n_time, 1);
    q_user_time = cell(n_time, 1);
    q_nonuser_time(:) = {[]};
    q_user_time(:) = {[]};
    m = cell(n_time, 1);
    n_user = cell(n_time, 1);
    n_nonuser = cell(n_time, 1);
    for iter_time=1:n_time
        q_time{iter_time, 1} = readmatrix(fullfile(inputFolder1, strcat('q', num2str(iter_time), fileName0, '.txt')));
        n_t = sum(q_time{iter_time, 1});
        n_nonuser_temp = round(nonuser_perc_prob*n_t);
%         n_user_temp = n_t - n_nonuser_temp;
        selected_nonusers = randsample(1:n_t,n_nonuser_temp);
        counter_driver = 0;
        if iter_time>=n_time_inc_start && iter_time<=n_time_inc_end
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
else
    fileName2 = fullfile(outputFolder2,...
                        strcat('data_', num2str(1), '.mat'));
    load(fileName2, 'q*', 'm', 'n_user', 'n_nonuser');
end

S_user = cell(n_time, 1);
S_nonuser = cell(n_time, 1);
v_userDet = cell(n_time, 1);
v_userDet(:) = {zeros(n_link*n_time, 1)};
% v_userStochastic = cell(n_time, 1);
% v_userStochastic(:) = {zeros(n_link*n_time, 1)};
v_nonuser = cell(n_time, 1);
v_nonuser(:) = {zeros(n_link*n_time, 1)};
% v_dynamic_array = zeros(n_link*n_time, 1); % users' volume of links
v_dynamic_array_NoInc_Det = zeros(n_link*n_time, 1); % users' volume of links  where company drivers act deterministic
% v_dynamic_array_NoInc_allStoch = zeros(n_link*n_time, 1); % users' volume of links where all drivers act stochastic
v_tilda_baseline = zeros(n_link*n_time, 1); % nonusers' volume of links
for iter_time=1:n_time
    S_user{iter_time} = zeros(m{iter_time, 1}, n_user{iter_time, 1});
    S_nonuser{iter_time} = zeros(m{iter_time, 1}, n_nonuser{iter_time, 1});
    counter_driver_user = 1;
    counter_driver_nonuser = 1;
    for iter_OD=1:n_OD
%         index_min = I(iter_OD);
        index_min = I_minTT{iter_time, 1}(iter_OD, 1); % !!!!!!!! iter_time=1 here <<>> Old setting
        n_user_time_temp = q_user_time{iter_time, 1}(iter_OD, 1);
        for iter_driver=1:n_user_time_temp
            S_user{iter_time}(index_min, counter_driver_user) = 1;
            counter_driver_user = counter_driver_user + 1;
        end
        n_nonuser_time_temp = q_nonuser_time{iter_time, 1}(iter_OD, 1);
        for iter_driver=1:n_nonuser_time_temp
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
    v_dynamic_array_NoInc_Det = v_dynamic_array_NoInc_Det + v_userDet{iter_time, 1};
%     v_dynamic_array_NoInc_allStoch = v_dynamic_array_NoInc_allStoch + v_userStochastic{iter_time, 1};
    v_tilda_baseline = v_tilda_baseline + v_nonuser{iter_time, 1};
end
clearvars  counter_driver_user counter_driver_nonuser index_min R_temp

%% Baseline travel time (Stochastic personal + Deterministic business)
% BPR constants
tt0_array = readmatrix(fullfile('data', 'capacity', region_, strcat('Mar2May_2018_new_5-22_link_tt_0_minutes_', region_, '.csv')));
w_array = readmatrix(fullfile('data', 'capacity', region_, strcat('Mar2May_2018_new_5-22_link_capacity_', region_, '.csv')));
% Length of links (miles)
L_array = readmatrix(fullfile('data', region_, strcat('link_length_mile_', region_, '_original.csv')));
% Volume function based on BPR function
% tt = @(x, v, tt0, w) (x+v)*tt0 + 0.15*tt0*((x+v)^5)/(w^4);
tt_single = @(x, v, tt0, w) tt0 + 0.15*tt0*((x+v)^4)/(w^4);
% tt_obj_NoInc_Det = zeros(n_link*n_time, 1);
% tt_obj_NoInc_Stoch = zeros(n_link*n_time, 1);
tt_obj_NoInc_Det_new = zeros(n_link*n_time, 1);
% tt_obj_NoInc_Stoch_new = zeros(n_link*n_time, 1);
% v_baseline = v_dynamic_array_NoInc_Det + v_tilda_baseline;
speed_link_time_new = zeros(n_link*n_time, 1);
volume_link_time_new = zeros(n_link*n_time, 1);

for iter_omega = 1:n_link*n_time
    tt0 = tt0_array(link_loc(mod(iter_omega-1, size(tt0_array, 1))+1, 1), 3);
    w = w_array(link_loc(mod(iter_omega-1, size(w_array, 1))+1, 1), 3);
    L = L_array(link_loc(mod(iter_omega-1, size(L_array, 1))+1, 1), 2);
    % v
    v_fixed_baseline = v_tilda_baseline(iter_omega, 1);
    v_dynamic_Det = v_dynamic_array_NoInc_Det(iter_omega, 1);
    % tt
    tt_temp_Det = tt_single(v_dynamic_Det, v_fixed_baseline, tt0, w);
    % speed
    speed_temp_Det = L/(tt_temp_Det/60);
    
    % Minute to hour
    speed_link_time_new(iter_omega, 1) = speed_temp_Det;  
    volume_link_time_new(iter_omega, 1) = v_fixed_baseline+v_dynamic_Det;
    tt_obj_NoInc_Det_new(iter_omega, 1) = tt_temp_Det*volume_link_time_new(iter_omega, 1);
end
% tt_obj_NoInc_Det_total = sum(tt_obj_NoInc_Det)/60; % Division by 60 to convert minutes to hours
% fprintf('Baseline travel time of the setting %s (hours): %.6f\n\n\n\n', setting_output, tt_obj_NoInc_Det_total)
tt_obj_NoInc_Det_total_new = sum(tt_obj_NoInc_Det_new)/60; % Division by 60 to convert minutes to hours
fprintf('New travel time of the setting %s (hours): %.6f\n\n\n\n', setting_output, tt_obj_NoInc_Det_total_new)

%% Initialize vector of companies
n_companies = 2;
if iterRun==1
    % q vectors of each company. It includes the # of drivers between ODs
    % size q_company{., .} = n_OD*n_time
    q_company = cell(n_companies, 1);
    q_company(:) = {[]};
    n_t = sum(q_user);
    n_c1_temp = round(1/n_companies*n_t);
    selected_users = randsample(1:n_t, n_c1_temp);
    counter_driver = 0;
    for iter_OD_time=1:size(q_user, 1)
        n_c1_OD_time = 0;
        for idx_driver = 1:q_user(iter_OD_time)
            counter_driver = counter_driver + 1;
            n_c1_OD_time = n_c1_OD_time + sum(ismember(selected_users, counter_driver));
        end
        q_company{1} = [q_company{1}; n_c1_OD_time];
        q_company{2} = [q_company{2}; q_user(iter_OD_time)-n_c1_OD_time];
        assert(q_company{1}(end)+q_company{2}(end)==q_user(iter_OD_time));
    end
    % Is (# if drivers) == (# of users)?
    q_sum_check=0;
    for iter_company=1:n_companies
        q_sum_check = q_sum_check + q_company{iter_company};
    end
    assert(sum(abs(q_user-q_sum_check))==0);
    clearvars n_user_temp_check q_sum_check
    % q vectors of each company at each time
    % size q_company_time{., .} = n_OD
    q_company_time = cell(n_companies, n_time);
    for iter_company=1:n_companies
        q_temp = q_company{iter_company};
        for iter_time=1:n_time
            q_company_time{iter_company, iter_time} = q_temp(1+(iter_time-1)*n_OD:iter_time*n_OD, 1);
        end
    end
    clearvars q_temp
    % # of drivers of each company
    % size n_driver_company{., .} = 1
    n_driver_company = cell(n_companies, 1);
    for iter_company=1:n_companies
        n_driver_company{iter_company} = sum(q_company{iter_company});
    end
else
    outputFolder2 = fullfile(folderRunDet, strcat('Run_', num2str(1)));
    fileName2 = fullfile(outputFolder2,...
                        strcat('data_', num2str(1), '.mat'));
    load(fileName2, 'q_company', 'q_company_time', 'n_driver_company');
end
%% Statistics of data
if iterRun==1
    fprintf('Number of drivers: %i\n', sum(q));
    fprintf('Number of user drivers: %i\n', sum(q_user));
    fprintf('Number of nonuser drivers: %i\n', sum(q_nonuser));
    for iter_company=1:n_companies
        fprintf('Number of drivers of company %i: %i\n', iter_company, n_driver_company{iter_company});
    end
end
%% Save data        
if iterRun==1
    mkdir(folderRunDet);
    outputFolder = fullfile(folderRunDet, strcat('Run_', num2str(iterRun), '_initEta'));
    mkdir(outputFolder);
    fileName = fullfile(outputFolder,...
                        strcat('data_', num2str(iterRun), '.mat'));
else
    outputFolder = fullfile(folderRunDet, strcat('Run_', num2str(iterRun), '_initEta'));
    mkdir(outputFolder);
    fileName = fullfile(outputFolder,...
                        strcat('data_', num2str(iterRun), '.mat'));
end
% Fix the idx of speed and volume data & save
volume_Det_reshaped = reshape(volume_link_time_new(:, 1), n_link, n_time);
volume_Det_reshaped_2save = volume_Det_reshaped(link_loc(:, 1), :);
dlmwrite(fullfile(outputFolder, 'vAftAsDet.csv'), volume_Det_reshaped_2save, 'precision', 10);
speed_Det_reshaped = reshape(speed_link_time_new(:, 1), n_link, n_time);
speed_Det_reshaped_2save = speed_Det_reshaped(link_loc(:, 1), :);
dlmwrite(fullfile(outputFolder, 'sAftAsDet.csv'), speed_Det_reshaped_2save, 'precision', 10);

save(fileName);    
