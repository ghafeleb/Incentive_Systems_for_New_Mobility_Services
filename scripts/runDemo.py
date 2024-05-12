import sys
import platform
if platform.system() == 'Windows':
    sys.path[0]+='\\..'
else:
    sys.path.insert(0, '../')
import argparse
from lib.utils import run_matlab, run_python, solver_name_to_n_iter, check_exists


def main(args):
    print("Run demo")

    # Create a graph with 2 nodes (Origin-Destination) and nPath paths between them. 
    # The same files will be generated for both py2 and py3 versions
    run_python(py_version = 2, 
           py_file = "../data_creator/createGraphPy2.py", 
           arg_names = ["config_filename", "nPath"], 
           arg_values = [args.config_filename, args.nPath])

    run_python(py_version = 3, 
        py_file = "../data_creator/createGraphPy3.py", 
        arg_names = ["config_filename", "nPath"], 
        arg_values = [args.config_filename, args.nPath])

    # 1. Find all the paths 2. OD estimation 3. Create files for ADMM (R, D, and travel times)
    run_python(py_version = 3, 
        py_file = "../data_creator/DataCreator1.py", 
        arg_names = ["config_filename"], 
        arg_values = [args.config_filename])

    # Compute speed and volume based on the data of graph
    run_python(py_version = 2, 
        py_file = "../data_creator/HistoricalData.py", 
        arg_names = ["config_filename"], 
        arg_values = [args.config_filename])


    for i in range(args.n_iter_UE):
        # ?
        run_matlab(matlab_function = "addpath('../data_creator/');DataLoader_Det", 
                    matlab_input_list = [i, \
                                        args.n_time, \
                                        args.n_time_inc_start, \
                                        args.n_time_inc_end, \
                                        args.seed_data, \
                                        args.step_size_UE, \
                                        args.region_, \
                                        args.setting_region, \
                                        args.fileName0, \
                                        args.folderName0, \
                                        ], 
                    nojvm = False,
                )
        # ?
        run_python(py_version = 2, 
                   py_file = "../data_creator/runDet.py", 
                   arg_names = ["config_filename", \
                                "sD", \
                                "iterRun", \
                                "initEta", \
                               ], 
                   arg_values = [args.config_filename, \
                                 args.seed_data, \
                                 i, \
                                 -1, \
                                ],
                   )

        # ?
        run_python(py_version = 3, 
                   py_file = "../data_creator/DataCreator2.py", 
                   arg_names = ["config_filename", \
                                "sD", \
                                "iterRun", \
                                "initEta", \
                                ],
                   arg_values = [args.config_filename, \
                                 args.seed_data, \
                                 i, \
                                 -1, \
                                 ],
                   )

    initEta_prestep_input_list = [args.n_iter_UE, \
                                  args.n_time, \
                                  args.n_time_inc_start, \
                                  args.n_time_inc_end, \
                                  args.seed_data, \
                                  args.step_size_UE, \
                                  args.region_, \
                                  args.setting_region, \
                                  args.fileName0, \
                                  args.folderName0]
    run_matlab(matlab_function = "addpath('../data_creator/');initEta_prestep", 
                matlab_input_list = initEta_prestep_input_list, 
                nojvm = False,
            )

    run_python(py_version = 2, 
           py_file = "../data_creator/runDet.py", 
           arg_names = ["config_filename", \
                        "sD", \
                        "iterRun", \
                        "initEta"], 
           arg_values = [args.config_filename, \
                         args.seed_data, \
                         args.n_iter_UE, \
                         1],
          )

    run_python(py_version = 3, 
               py_file = "../data_creator/DataCreator2.py", 
               arg_names = ["config_filename", \
                            "sD", \
                            "iterRun", \
                            "initEta"],  
               arg_values = [args.config_filename, \
                             args.seed_data, \
                             args.n_iter_UE, \
                             1])
    
    
    
    
    
    
    
    n_iter_algo = args.n_iter_ADMM if args.solver_name == "ADMM" else solver_name_to_n_iter(args.solver_name)
    if args.solver_name == "ADMM":
        # Run Algorithm 1
        ADMM_input_list = [args.n_iter_UE, \
                           args.percNonUVal, \
                           args.budget, \
                           args.n_iter_ADMM, \
                           args.n_time, \
                           args.n_time_inc_start, \
                           args.n_time_inc_end, \
                           args.VOT, \
                           args.seed_solver, \
                           args.seed_data, \
                           args.rho, \
                           args.fairness, \
                           args.step_size_UE, \
                           args.region_, \
                           args.setting_region, \
                           args.fileName0, \
                           args.folderName0]
        run_matlab(matlab_function = "addpath('../incentivization/');ADMM", 
                    matlab_input_list = ADMM_input_list, 
                    nojvm = True,
            )

        # Find binary solution to the provided solution by Algorithm 1 
        ILP_input_list = [args.n_iter_UE, \
                           args.percNonUVal, \
                           args.budget, \
                           args.n_iter_ADMM, \
                           args.n_time, \
                           args.n_time_inc_start, \
                           args.n_time_inc_end, \
                           args.VOT, \
                           args.seed_solver, \
                           args.seed_data, \
                           args.rho, \
                           args.fairness, \
                           args.step_size_UE, \
                           args.region_, \
                           args.setting_region, \
                           args.MIPGap]
        run_matlab(matlab_function = "addpath('../incentivization/');ILP", 
                    matlab_input_list = ILP_input_list, 
                    nojvm = True,
            )
    else:
        # Solve the MIP optimization problem directly using a solver
        solver_input_list = [args.solver_name, \
                                args.MIPGap, \
                                args.solve_binarization, \
                                args.n_time, \
                                args.n_time_inc_start, \
                                args.n_time_inc_end, \
                                args.seed_data, \
                                args.seed_solver, \
                                args.percNonUVal, \
                                args.budget, \
                                args.VOT, \
                                args.fairness, \
                                args.n_iter_UE, \
                                args.step_size_UE, \
                                args.region_, \
                                args.setting_region, \
                                args.fileName0, \
                                args.folderName0]
        run_matlab(matlab_function = "addpath('../incentivization/');solver_incentivization", 
                    matlab_input_list = solver_input_list, 
                    nojvm = True,
                )
        
        
        
    # Get the volume and speed based on the post-incentivization traffic
    getSV_realCost_input_list = [args.n_iter_UE, \
                                 args.percNonUVal, \
                                 args.budget, \
                                 args.n_companies_ADMM, \
                                 n_iter_algo, \
                                 args.n_time, \
                                 args.n_time_inc_start, \
                                 args.n_time_inc_end, \
                                 args.fairness, \
                                 args.VOT, \
                                 args.seed_solver, \
                                 args.seed_data, \
                                 args.rho, \
                                 args.step_size_UE,\
                                 args.region_, \
                                 args.setting_region, \
                                 args.MIPGap, ]
    run_matlab(matlab_function = "addpath('../incentivization/');getSV_realCost", 
                matlab_input_list = getSV_realCost_input_list, 
                nojvm = False,
            )
        
    # Compute the R matrix and other info based on new speed and volume
    run_python(py_version = 2, 
               py_file = "../incentivization/run_realCost.py", 
               arg_names = ["config_filename", \
                            "sA", \
                            "sD", \
                            "nC", \
                            "f", \
                            "percNonU", \
                            "iterRun", \
                            "b", \
                            "VOT", \
                            "nTIS", \
                            "nTIE", \
                            "it", \
                            "MIPGap", \
                           ], 
               arg_values = [args.config_filename, \
                             args.seed_solver, \
                             args.seed_data, \
                             args.n_companies_ADMM, \
                             args.fairness, \
                             args.percNonUVal, \
                             args.n_iter_UE, \
                             args.budget, \
                             args.VOT, \
                             args.n_time_inc_start, \
                             args.n_time_inc_end, \
                             n_iter_algo, \
                             args.MIPGap, \
                            ],
              )
        
    # Compute the incentivization cost
    compareCosts_realCost_initAll2_input_list = [args.n_iter_UE, \
                                                 args.percNonUVal, \
                                                 args.budget, \
                                                 args.n_companies_ADMM, \
                                                 n_iter_algo, \
                                                 args.min_n_companies, \
                                                 args.max_n_companies, \
                                                 args.step_n_companies, \
                                                 args.factor_n_companies, \
                                                 # Number of samples of cost computation, randomness is in selecting companies' drivers
                                                 args.n_sample, \
                                                 args.n_time, \
                                                 args.n_time_inc_start, \
                                                 args.n_time_inc_end, \
                                                 args.fairness, \
                                                 args.VOT, \
                                                 args.seed_solver, \
                                                 args.seed_data, \
                                                 args.rho, \
                                                 args.step_size_UE,\
                                                 args.region_, \
                                                 args.setting_region, \
                                                 args.MIPGap, \
                                                ]
    run_matlab(matlab_function = "addpath('../incentivization/');compareCosts_realCost_initAll2_allInOne", 
                matlab_input_list = compareCosts_realCost_initAll2_input_list, 
                nojvm = True,
            )
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_', default="region_toy", type=str, help='Region name')
    parser.add_argument('--config_filename', default="../data/YAML/region_toy.yaml", type=str, help='Config of the data.')
    parser.add_argument('--nPath', default=2, type=int, help='Number of paths of graph.')
    parser.add_argument('--n_iter_UE', default=5, type=int, help='Number of iterations of UE algorithm for convergence.')
    parser.add_argument('--n_iter_ADMM', default=100, type=int, help='Number of iterations of ADMM based algorithm.')
    parser.add_argument('--step_size_UE', default=0.01, type=float, help='Step size of the UE algorithm')
    parser.add_argument('--percNonUVal', default=50, type=int, help='Percentage of nonuser drivers')
    parser.add_argument('--n_time', default=204, type=int, help='Number of time buckets in the data')
    parser.add_argument('--n_time_inc_start', default=13, type=int, help='Index of the starting time bucket for the analysis')
    parser.add_argument('--n_time_inc_end', default=24, type=int, help='Index of the ending time bucket for the analysis')
    parser.add_argument('--setting_region', default="5_22_AVG5_th1_pad_MultipleTimes", type=str, help='Setting name of the region data.')
    parser.add_argument('--fileName0', default="_StartHour_5_AVG5_pad_theta1e+00", type=str, help='Suffix of data names from OD estimation algorithm.')
    parser.add_argument('--folderName0', default="Mar2May_2018_new_5-22_", type=str, help='Prefix of folder names of the data setting.')
    parser.add_argument('--fairness', default="0_0_0_100_0", type=str, help='Percentage of groups of drivers with the corresponding fairness multipier from the list [x1.0, x1.1, x1.5, x2.0, x2.5]')
    parser.add_argument('--seed_data', default=2, type=int, help='Seed of UE algorithm.')
    parser.add_argument('--seed_solver', default=2, type=int, help='Seed of solver.')
    parser.add_argument('--budget', default=10000, type=int, help='incentivization budget.')
    parser.add_argument('--rho', default=20, type=int, help='ADMM hyperparameter')
    parser.add_argument('--n_companies_ADMM', default=1, type=int, help='Index of the starting time bucket for the analysis')
    parser.add_argument('--n_companies_cost', default=2, type=int, help='Index of the ending time bucket for the analysis')
    parser.add_argument('--VOT', default=2.63, type=float, help='Value of Time ($) per minute')
    parser.add_argument('--MIPGap', default=0.01, type=float, help='Accuracy of the solver')
    parser.add_argument('--solver_name', default="ADMM", type=str, help='Solver of the optimization problems')
    parser.add_argument('--min_n_companies', default=1, type=int, help='The min number of organizations in the cost computation')
    parser.add_argument('--max_n_companies', default=10000, type=int, help='The max number of organizations in the cost computation')
    parser.add_argument('--step_n_companies', default=10, type=int, help='Setting 1: number of organization is range(min_n_companies, step_n_companies, max_n_companies) + {# of drivers OR individual drivers}. Setting 1 is set if factor_n_companies=0. Setting 1 is much more time consuming but more details comapred to setting 2.')
    parser.add_argument('--factor_n_companies', default=10, type=int, help='Setting 2: number of organization is {min_n_companies, min_n_companies*factor_n_companies, ..., max_n_companies, # of drivers OR individual drivers}.')
    parser.add_argument('--n_sample', default=10, type=int, help='Number of random sampling of drivers to companies and computing cost of incentivization.')
    parser.add_argument('--solve_binarization', default=1, type=int, help='For solvers other than Algorithm 1: 1=True, 0=False.')
    args = parser.parse_args()
    main(args)
