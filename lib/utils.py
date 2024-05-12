import sys
import os
# import platform
# if platform.system() == 'Windows':
#     sys.path[0]+='\\..'
# else:
#     sys.path.insert(0, '../')
# print(platform.system())
import subprocess
from itertools import chain


def check_exists(add_, create_=True):
    if not os.path.exists(add_):
        print('Does not exist: \n' + add_)
        if create_:
            print('Creating the folder...')
            os.mkdir(add_)
            
def solver_name_to_n_iter(solver_name):
    if solver_name == "Gurobi":
        return -1
    elif solver_name == "Mosek":
        return -2
    elif solver_name == "GLPK":
        return -3
    return -4

def get_solution_result_folder_address(budget, \
                                    seed_data, \
                                    seed_solving_algo,\
                                    VOT,\
                                    fairness,\
                                    percNonUVal,\
                                    n_time_inc_start,\
                                    n_time_inc_end,\
                                    step_size,\
                                    iterRun,\
                                     region_,\
                                     setting_region,\
                                     solver_name,\
                                    n_iter_ADMM = None, \
                                    rho = None,):
    iterRun += 1
    parentAddress = os.path.join('..', 'data', region_, setting_region)
    if solver_name == "ADMM":
        folderName = "".join(['Det_initAll2_MultT', \
                                '_b', str(budget), \
                                '_sD', str(seed_data), \
                                '_sA', str(seed_solving_algo), \
                                '_r', str(rho), \
                                '_it', str(n_iter_ADMM), \
                                '_VOT', str(VOT), \
                                '_nC', str(1), \
                                '_f', fairness, \
                                '_initSB_T', \
                                '_percNonU', str(percNonUVal), \
                                '_nTIS', str(n_time_inc_start), \
                                '_nTIE', str(n_time_inc_end), \
                                '_ss', str(step_size), \
                                '_itN', str(iterRun)])
#         fileName = 'result_'+ str(iterRun) + '.mat'
        folderAddress = os.path.join(parentAddress, folderName)
        
    else:
        folderName = "".join([solver_name, \
                            '_new_Det_initAll2_MultT', \
                                    '_b', str(budget), \
                                    '_sD', str(seed_data), \
                                    '_sS', str(seed_solving_algo), \
                                    '_VOT', str(VOT), \
                                    '_nC', str(1), \
                                    '_f', fairness, \
                                    '_percNonU', str(percNonUVal), \
                                    '_nTIS', str(n_time_inc_start), \
                                    '_nTIE', str(n_time_inc_end), \
                                    '_ss', str(step_size), \
                                    '_itN', str(iterRun)])

#         fileName = solver_name + '_solver_result.mat'
        folderAddress = os.path.join(parentAddress, folderName)
        
    return folderAddress

def get_solution_result_file_address(solver_name, MIPGap):
    if solver_name == "ADMM":
        fileName = 'result_MIPGap' + str(MIPGap) + '_ILP.mat' 
    else:
        fileName = solver_name + '_MIPGap' + str(MIPGap) + '_solver_result.mat'
    return fileName

def run_matlab(matlab_function, matlab_input_list, nojvm=False):	
    # print('\nRunning matlab file:', matlab_function)
    matlab_input_list_modified = []
    for x in matlab_input_list:
        if type(x) == str:
            matlab_input_list_modified.append("'" + x + "'")
        else:
            matlab_input_list_modified.append(str(x))
    matlab_input = ','.join(x for x in matlab_input_list_modified)
    command = "".join(["try;", matlab_function, "(", matlab_input, ");catch;end;quit"])
    # command = "".join(["try;", matlab_function, "(", matlab_input, ");catch;end"])
    print('Running command: ', command)
    if nojvm:
        process = subprocess.Popen(['matlab', '-r', '-wait', command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        process = subprocess.Popen(['matlab', '-r', '-wait', '-nodesktop', '-nojvm', command], shell=True)
    process.wait()
    # print('Finished running matlab file:', matlab_function)


def run_python(py_version, py_file, arg_names, arg_values):	
    # print('\nRunning Python', str(py_version),' file: ', py_file)
    if arg_names:
        dashed_arg_names = ['--'+arg_name for arg_name in arg_names]
        arg_values = [str(x) for x in arg_values]
        py_input = list(chain.from_iterable(zip(dashed_arg_names, arg_values)))
    else:
        py_input = []
    command = " ".join(["python"+str(py_version), py_file] + py_input)
    print('Running command: ', command)
    process = subprocess.Popen(["python"+str(py_version), py_file] + py_input, shell=True)
    process.wait()
    # print('Finished running Python', str(py_version),' file: ', py_file)


def O_list_D_list(nPath, ODNodes):
    # O_list = [1, 1, 2, 3]
    O_list= []
    for iPath in range(nPath):
        O_list.append(1)
        O_list.append(iPath+2)
    # D_list = [2, 3, 4, 4]
    D_list = []
    for i in range(nPath):
        D_list.append(i+2)
        D_list.append(nPath+2)

    return O_list, D_list


def od_list_f(nPath, ODNodes, region_):
    # if region_ == 'region_toy2':
    #     od_list = [ODNodes, ODNodes]
    # elif region_ == 'region_toy2_1OD' or region_ == 'region_toy2_1OD_defODEst_2' or region_ == 'region_toy2_1OD_defODEst_5min':
    od_list = [[1], [nPath+2]]
    return od_list


def AM_PM_f(t):
    if t > 12:
        return str(t - 12) + 'PM'
    elif t == 12:
        return '12PM'
    else:
        return str(t) + 'AM'


def hr_str(start_hr, finish_hr, AM_PM):
    if AM_PM:
        return AM_PM_f(start_hr), AM_PM_f(finish_hr)
    else:
        return str(start_hr), str(finish_hr)

def mph2kph(s_0_mph):
    s_0_kph = s_0_mph * 1.60934
    return s_0_kph

def mph2kph_dict(s_0_mph):
    s_0_kph = [s_0_mph[iter] * 1.60934 for iter in s_0_mph.keys()]
    # s_0_kph = [s_0_mph[iter] * 1.60934 for iter in range(len(s_0_mph))]
    return s_0_kph

def m2km(mile_):
    km_ = [mile_[iter] * 1.60934 for iter in range(len(mile_))]
    return km_
