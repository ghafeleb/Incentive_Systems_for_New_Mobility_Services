import sys
from lib.utils import check_exists, run_python, run_matlab, get_solution_result_folder_address, \
                        get_solution_result_file_address
import os
import scipy.io
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
import numpy as np

def get_optimization_solution(fileAddress, \
                              solver_name, \
                             ):
    # Load variables from solution of the optimization model
#     if solver_name == "ADMM" and VOT != 2.63 and percNonUVal != 90:
#         continue
    variables2loadSolver = ["tt_obj_total_NoInc_7_830", "tt_obj_7_830"]
    if solver_name == "ADMM":
        variables2loadSolver.append("nDeviated1")
    variablesDictSolver = scipy.io.loadmat(file_name=fileAddress,  variable_names=variables2loadSolver)
    tt_obj_total_NoInc_7_830 = variablesDictSolver["tt_obj_total_NoInc_7_830"].item()
    tt_obj_7_830 = variablesDictSolver["tt_obj_7_830"].item()
    n_deviated = 0
    if solver_name == "ADMM":
        n_deviated = variablesDictSolver['nDeviated1'].item()
    return tt_obj_total_NoInc_7_830, tt_obj_7_830, n_deviated

def get_cost(folderAddress, factor_cost_comparison, MIPGap):
    # Load variables from incentivization cost
#     if VOT != 2.63:
#         continue
    fileNameCost = 'compare_costs_nC_factor' + str(factor_cost_comparison) + '.mat'
    fileAddressCost = os.path.join(folderAddress, 'cost_MIPGap' + str(MIPGap) , fileNameCost)
    variables2loadSolverCost = ["cost_summary"]
    variablesDictSolverCost = scipy.io.loadmat(file_name=fileAddressCost,  variable_names=variables2loadSolverCost)
#     solver_cost_dict[(solver_name, VOT, percNonUVal, budget)] = variablesDictSolverCost['cost_summary']
    solver_cost_summary = variablesDictSolverCost['cost_summary']
    solver_cost_dict = {}
    for idx_n_company in range(len(solver_cost_summary)):
        n_company = solver_cost_summary[idx_n_company][0].item() 
        solver_cost_dict[n_company] = solver_cost_summary[idx_n_company][1].item()
#     n_companies = [solver_cost_summary[nC][0].item() for nC in range(len(solver_cost_summary))]
#     solver_cost_dict = {}
#     for idx, nC in enumerate(n_companies):
#         solver_cost_dict[nC] = solver_cost_summary[idx][1].item()
    return solver_cost_dict

def get_budget_cost_plot_dict(solver_cost_dict, solver_list, VOT_list, percNonUVal_list, budget_list):
    budget_cost_plot_dict = {}
    for solver_name, VOT, percNonUVal in itertools.product(solver_list, VOT_list, percNonUVal_list):
        budget_cost_plot_dict[(solver_name, VOT, percNonUVal)] = {}
        for budget in budget_list:
            for n_company, avg_cost in solver_cost_dict[(solver_name, VOT, percNonUVal, budget)].items():
                avg_cost = round(avg_cost, 2)
                # Data for budget-cost reduction plot
                if n_company in budget_cost_plot_dict[(solver_name, VOT, percNonUVal)]:
                    budget_cost_plot_dict[(solver_name, VOT, percNonUVal)][n_company][0].append(int(budget))
                    budget_cost_plot_dict[(solver_name, VOT, percNonUVal)][n_company][1].append(avg_cost)
                else:
                    budget_cost_plot_dict[(solver_name, VOT, percNonUVal)][n_company] = [[0, int(budget)], \
                                                                                         [0, avg_cost]]
    return budget_cost_plot_dict

def get_budget_tt_reduction_plot_dict(solver_solution_dict, solver_list, VOT_list, percNonUVal_list, budget_list):
    budget_tt_reduction_plot_dict = {}
    for solver_name, VOT, percNonUVal in itertools.product(solver_list, VOT_list, percNonUVal_list):
        budget_tt_reduction_plot_dict[(solver_name, VOT, percNonUVal)] = [[0], [0]]
        for budget in budget_list:
            # Data for budget-tt reduction plot
            budget_tt_reduction_plot_dict[(solver_name, VOT, percNonUVal)][0].append(budget)
            tt_decrease_perc = solver_solution_dict[(solver_name, VOT, percNonUVal, budget)]["tt_decrease_perc"]
            budget_tt_reduction_plot_dict[(solver_name, VOT, percNonUVal)][1].append(tt_decrease_perc)
    return budget_tt_reduction_plot_dict

def get_budget_cost_per_deviated_plot_dict(n_deviated_dict, solver_cost_dict, solver_list, VOT_list, percNonUVal_list, budget_list):
    budget_cost_per_deviated_plot_dict = {}
    for solver_name, VOT, percNonUVal in itertools.product(solver_list, VOT_list, percNonUVal_list):
        budget_cost_per_deviated_plot_dict[(solver_name, VOT, percNonUVal)] = {}
    #     print("\n")
        for budget in budget_list:
            if solver_name == "ADMM":
                # Data for budget-n deviated plot
                n_deviated = n_deviated_dict[(solver_name, VOT, percNonUVal, budget)]
            for n_company, avg_cost in solver_cost_dict[(solver_name, VOT, percNonUVal, budget)].items():
                avg_cost = round(avg_cost, 2)
                # Data for cost per deviated driver-budget plot
                if solver_name == "ADMM":
                    if n_company in budget_cost_per_deviated_plot_dict[(solver_name, VOT, percNonUVal)]:
                        budget_cost_per_deviated_plot_dict[(solver_name, VOT, percNonUVal)][n_company][0].append(int(budget))
                        budget_cost_per_deviated_plot_dict[(solver_name, VOT, percNonUVal)][n_company][1].append(avg_cost/n_deviated)
                    else:
                        budget_cost_per_deviated_plot_dict[(solver_name, VOT, percNonUVal)][n_company] = [[0, int(budget)], \
                                                                                                          [0, avg_cost/n_deviated]]
    return budget_cost_per_deviated_plot_dict

def get_cost_tt_reduction_plot_dict(solver_cost_dict, solver_solution_dict, solver_list, VOT_list, percNonUVal_list, budget_list):
    cost_tt_reduction_plot_dict = {}
    for solver_name, VOT, percNonUVal in itertools.product(solver_list, VOT_list, percNonUVal_list):
        cost_tt_reduction_plot_dict[(solver_name, VOT, percNonUVal)] = {}
        for budget in budget_list:
            tt_decrease_perc = solver_solution_dict[(solver_name, VOT, percNonUVal, budget)]['tt_decrease_perc']
            for n_company, avg_cost in solver_cost_dict[(solver_name, VOT, percNonUVal, budget)].items():
                avg_cost = round(avg_cost, 2)
                # Data for cost-tt reduction plot
                if n_company in cost_tt_reduction_plot_dict[(solver_name, VOT, percNonUVal)]:
                    cost_tt_reduction_plot_dict[(solver_name, VOT, percNonUVal)][n_company][0].append(avg_cost)
                    cost_tt_reduction_plot_dict[(solver_name, VOT, percNonUVal)][n_company][1].append(tt_decrease_perc)
                else:
                    cost_tt_reduction_plot_dict[(solver_name, VOT, percNonUVal)][n_company] = [[0, avg_cost], \
                                                                                               [0, tt_decrease_perc]]
    return cost_tt_reduction_plot_dict

def get_n_deviated_plot_dict(n_deviated_dict, solver_list, VOT_list, percNonUVal_list, budget_list):
    n_deviated_plot_dict = {}
    for solver_name, VOT, percNonUVal in itertools.product(solver_list, VOT_list, percNonUVal_list):
        n_deviated_plot_dict[(solver_name, VOT, percNonUVal)] = [[0], [0]]
        for budget in budget_list:
            if solver_name == "ADMM":
                # Data for budget-n deviated plot
                n_deviated_plot_dict[(solver_name, VOT, percNonUVal)][0].append(int(budget))
                n_deviated = n_deviated_dict[(solver_name, VOT, percNonUVal, budget)]
                n_deviated_plot_dict[(solver_name, VOT, percNonUVal)][1].append(int(n_deviated))
    return n_deviated_plot_dict

def plot_tt_reduction_perc_new(data_dict, solver_names, percNonUVals, VOTs, blueBack, large_size=False, fontsize=28):
    if large_size:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))
    for percNonUVal_iter in percNonUVals:
        for (solver_name_temp, VOT, percNonUValTemp), y_axis_list in data_dict.items():
            if percNonUValTemp != percNonUVal_iter:
                    continue
            if solver_name_temp not in solver_names or VOT not in VOTs:
                continue
            if solver_name_temp == "ADMM":
                solver_name_algo = "Algorithm 1"
            else:
                solver_name_algo = solver_name_temp
            # x-axis
            x_axis_str = []
            for val in data_dict[(solver_name_temp, VOT, percNonUValTemp)][0]:
                if val == 0:
                    x_axis_str.append('$0,\nNo Incentive')
                else:
                    x_axis_str.append('$'+str(val))
            # Plot label
            percU = 100 - percNonUValTemp
            if large_size:
                label_str = str(percU) + "%" + ", " + str(VOT) + ", " + solver_name_algo
            else:
                if len(solver_names) > 1 and len(VOTs) > 1:
                    label_str = str(percU) + "%" + ", "  + solver_name_algo + ", " + str(VOT)
                elif len(VOTs) > 1:
                    label_str = str(percU) + "%" + ", " + "VOT=" + str(VOT)
                elif len(solver_names) > 1:
                    label_str =  str(percU) + "%" + ", " + solver_name_algo
                else:
                    label_str = str(percU) + "%"
            # Plot 
            if large_size:
                ax.plot(x_axis_str, data_dict[(solver_name_temp, VOT, percNonUValTemp)][1], \
                    linewidth=8, marker='o', markersize=14, label=label_str)
            else:
                ax.plot(x_axis_str, data_dict[(solver_name_temp, VOT, percNonUValTemp)][1], \
                    linewidth=16, marker='o', markersize=28, label=label_str)
#     ax.legend()
    if large_size:
        ax.legend(title='Penetration Rate, Solver, VOT')
    else:
        if len(solver_names) > 1 and len(VOTs) > 1:
            legend_title = 'Penetration Rate, Solver, VOT'
        elif len(VOTs) > 1:
            legend_title = 'Penetration Rate, VOT'
        elif len(solver_names) > 1:
            legend_title = 'Penetration Rate, Solver'
        else:
            legend_title = 'Penetration Rate'
        legend = plt.legend(title=legend_title, fontsize=fontsize-4,    # Smaller text
                                           markerscale=0.5,       # Smaller markers relative to plot
                                           handlelength=2,        # Shorter handles
                                           handleheight=1,        # Smaller handle height
                                           borderpad=0.5,         # Smaller border padding
                                           labelspacing=0.5,      # Less space between labels
                                           handletextpad=0.5)     # Less space between handle and text)
        legend.get_title().set_fontsize(fontsize-4)  # Set smaller font size for the legend title
        
    # plt.legend(title='Penetration Rate (%)').remove()
#     plt.title(', '.join([solve_method, result_to_plot, 'tt_reduction_perc', args.region_]))
    ax.set_ylabel('Travel Time Reduction', fontsize=fontsize)
#     plt.ylim(0.0, 9.0)
    ax.set_xlabel('Budget', fontsize=fontsize)
    fmt = '{x:,.2f}%'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    ax.tick_params(axis='both', labelsize=fontsize)
    
    plt.tight_layout()        
    
    # Plot name
    plot_name = 'ttReductionPerc' + \
                 '_VOT' + "_".join([str(x) for x in VOTs]) + \
                 '_solv' +  "_".join([str(x) for x in solver_names]) + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVals])
    
    # Blue background for rebuttal
    if blueBack:
#         plt.rcParams['axes.facecolor'] = 'skyblue' 
        ax.set_facecolor('skyblue')  # Set subplot background color
        plot_name += '_blue'
    else:
#         plt.rcParams['axes.facecolor'] = 'white'
        ax.set_facecolor('white')  # Set subplot background color
    if large_size:
        plot_name = '12X6_' + plot_name
    # Save the plot
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)
    
    plt.show()    

def plot_tt_reduction_perc_new_subplots(ax, 
                                        data_dict, 
                                        solver_names, 
                                        percNonUVal, 
                                        VOTs, 
                                        blueBack, 
                                        large_size=False, 
                                        fontsize=28):
    for (solver_name_temp, VOT, percNonUValTemp), y_axis_list in data_dict.items():
        if percNonUValTemp != percNonUVal or solver_name_temp not in solver_names or VOT not in VOTs:
            continue
        if solver_name_temp == "ADMM":
            solver_name_algo = "Algorithm 1"
        else:
            solver_name_algo = solver_name_temp
        # x-axis
        x_axis_str = []
        for val in data_dict[(solver_name_temp, VOT, percNonUValTemp)][0]:
            if val == 0:
                x_axis_str.append('$0,\nNo Incentive')
            else:
                x_axis_str.append('$'+str(val))
        # Plot label
        percU = 100 - percNonUValTemp
        if large_size:
            label_str = solver_name_algo + ", " + str(VOT) + ", " + str(percU) + "%"
        else:
            label_str = solver_name_algo
        # Plot 
        if large_size:
            ax.plot(x_axis_str, data_dict[(solver_name_temp, VOT, percNonUValTemp)][1], \
                linewidth=8, marker='o', markersize=14, label=label_str)
        else:
            ax.plot(x_axis_str, data_dict[(solver_name_temp, VOT, percNonUValTemp)][1], \
                linewidth=16, marker='o', markersize=28, label=label_str)
#     ax.legend()
    if large_size:
        legend_title='Solver, VOT, Penetration Rate'
    else:
        legend_title = 'Solver'
        
        legend = ax.legend(title=legend_title, 
                           fontsize=fontsize-4,           # Smaller text
                           markerscale=0.5,       # Smaller markers relative to plot
                           handlelength=2,        # Shorter handles
                           handleheight=1,        # Smaller handle height
                           borderpad=0.5,         # Smaller border padding
                           labelspacing=0.5,      # Less space between labels
                           handletextpad=0.5,     # Less space between handle and text)
                           loc='lower right')
#                            loc='upper left')
        legend.get_title().set_fontsize(fontsize-4)
    
    # plt.legend(title='Penetration Rate (%)').remove()
#     plt.title(', '.join([solve_method, result_to_plot, 'tt_reduction_perc', args.region_]))
    ax.set_ylabel('Travel Time Reduction', fontsize=fontsize)
    ax.set_ylim(0.0, 9.0)
    ax.set_xlabel('Budget', fontsize=fontsize)
    fmt = '{x:,.2f}%'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    # Set font size for tick labels on both axes
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()        
    
def plot_tt_reduction_perc_new_subplots_complete(budget_tt_reduction_plot_dict,
                                                 solver_names_setting,
                                                 percNonUVals_setting,
                                                 VOTs_setting,
                                                blueBack = True,
                                                largesize = False,
                                                fontsize = 28,
                                                ):
    # Create a 2x2 grid of subplots
#     fig, axs = plt.subplots(2, 2, figsize=(16, 14), facecolor='skyblue')  # figsize is optional
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))  # figsize is optional

    # Flatten the array of Axes to make indexing easier
    axs = axs.flatten()

    # Call the function on each subplot
    for idx_plot, percNonUVal in enumerate(percNonUVals_setting):
        plot_tt_reduction_perc_new_subplots(ax = axs[idx_plot], \
                                   data_dict = budget_tt_reduction_plot_dict, \
                                   solver_names = solver_names_setting, \
                                   percNonUVal = percNonUVal, \
                                   VOTs = VOTs_setting, \
                                   blueBack = blueBack, 
                                   large_size = largesize,
                                   fontsize = fontsize,)
        if blueBack:
            axs[idx_plot].set_facecolor('skyblue')  # Set subplot background to blue
        else:
            axs[idx_plot].set_facecolor('white')  # Set subplot background to white
        
    # Label each subplot
    labels = ['(a) 5% Penetration Rate', \
              '(b) 10% Penetration Rate', \
              '(c) 15% Penetration Rate', \
              '(d) 20% Penetration Rate']
#     positions = [(0.95, 0.05), (0.95, 0.05), (0.95, 0.05), (0.95, 0.05)]
#     alignments = [('right', 'bottom'), ('right', 'bottom'), ('right', 'bottom'), ('right', 'bottom')]
    positions = [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95), (0.05, 0.95)]
    alignments = [('left', 'top'), ('left', 'top'), ('left', 'top'), ('left', 'top')]
    for ax, label, pos, align  in zip(axs.flatten(), labels, positions, alignments):
        # Position the text in the top left of the subplot
        # Adjust coordinates and alignment as needed
        ax.text(pos[0], pos[1], label, transform=ax.transAxes, fontsize=fontsize+4, fontweight='bold', va=align[1], ha=align[0])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Plot name
    plot_name = 'ttReductionPercAllInOne' + \
                 '_VOT' + "_".join([str(x) for x in VOTs_setting]) + \
                 '_solv' +  "_".join([str(x) for x in solver_names_setting]) + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVals_setting])

    # Blue background for rebuttal
    # plt.rcParams['axes.facecolor'] = 'white'
    if blueBack:
    #     plt.rcParams['axes.facecolor'] = 'skyblue' 
        plot_name += '_blue'
    if largesize:
        plot_name = '12X6_' + plot_name

    # Plot name + Save
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)

    # Display the figure
    plt.show()
    
def plot_tt_reduction_perc_comparison(data_dict, 
                                      solver_name1, 
                                      solver_name2, 
                                      percNonUVals, 
                                      VOTs, 
                                      blueBack, 
                                      large_size=False,
                                      perc_diff = False,
                                      fontsize = 20,
                                     ):
    if large_size:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))
    for (solver_name_temp, VOT, percNonUValTemp), (x_axis, cost_1) in data_dict.items():
        if percNonUValTemp not in percNonUVals or solver_name_temp != solver_name1 or VOT not in VOTs:
            continue
        if solver_name1 == "ADMM":
            solver_name_algo1 = "Algorithm 1"
        else:
            solver_name_algo1 = solver_name1
        if solver_name2 == "ADMM":
            solver_name_algo2 = "Algorithm 1"
        else:
            solver_name_algo2 = solver_name2
        # x-axis
        x_axis_str = []
        for val in x_axis:
            if val == 0:
                x_axis_str.append('$0,\nNo Incentive')
            else:
                x_axis_str.append('$'+str(val))
        # Plot label
        percU = 100 - percNonUValTemp
        if large_size:
            label_str = solver_name_algo1 + " vs " + solver_name_algo2 + ", " + str(VOT) + ", " + str(percU) + "%"
        else:
            if len(VOTs) > 1:
                label_str = "VOT=" + str(VOT) + ", " + str(percU) + "%"
            else:
                label_str = str(percU) + "%"
#         print(cost_1)
        cost_2 = data_dict[(solver_name2, VOT, percNonUValTemp)][1]
#         print(cost_2)
        assert len(cost_1)==len(cost_2), print("Number of costs are not equal")
        y_axis = []
        if perc_diff:
            for i in range(len(cost_1)):
                if cost_2[i] == 0:
                    y_axis.append(0)
                else:
                    y_axis.append((cost_1[i] - cost_2[i])/cost_2[i]*100)
        else:
            for i in range(len(cost_1)):
                if cost_2[i] == 0:
                    y_axis.append(0)
                else:
                    y_axis.append(cost_1[i] - cost_2[i])
        # Plot 
        if large_size:
            ax.plot(x_axis_str, y_axis, \
                linewidth=8, marker='o', markersize=14, label=label_str)
        else:
            ax.plot(x_axis_str, y_axis, \
                linewidth=16, marker='o', markersize=28, label=label_str)

    if large_size:
        legend_title='Solver, VOT, Penetration Rate'
    else:
        if len(VOTs) > 1:
            legend_title='VOT, Penetration Rate'
        else:
            legend_title='Penetration Rate'
    
    legend = ax.legend(title=legend_title, 
                       fontsize=fontsize-4,           # Smaller text
                       markerscale=0.5,       # Smaller markers relative to plot
                       handlelength=2,        # Shorter handles
                       handleheight=1,        # Smaller handle height
                       borderpad=0.5,         # Smaller border padding
                       labelspacing=0.5,      # Less space between labels
                       handletextpad=0.5)     # Less space between handle and text)
    legend.get_title().set_fontsize(fontsize-4)
    
#     plt.title(', '.join([solve_method, result_to_plot, 'tt_reduction_perc', args.region_]))
    solver_name_algo1 = "Algorithm 1" if solver_name1 == "ADMM" else solver_name1
    solver_name_algo2 = "Algorithm 1" if solver_name2 == "ADMM" else solver_name2
    ax.set_ylabel('Travel Time Reduction\n(' + solver_name_algo1 + ' - ' + solver_name_algo2 + ')', fontsize = fontsize)
#     plt.ylim(0.0, 9.0)
    ax.set_xlabel('Budget', fontsize = fontsize)
    fmt = '{x:,.2f}%'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    # Set font size for tick labels on both axes
    ax.tick_params(axis='both', labelsize=fontsize)
    
    plt.tight_layout()        
    
    # Plot name
    plot_name = 'ttReductionPerc' + \
                 '_VOT' + "_".join([str(x) for x in VOTs]) + \
                 '_solv1' +  solver_name1 + \
                 '_solv2' +  solver_name2 + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVals])
    
    # Blue background for rebuttal
    
    if blueBack:
#         plt.rcParams['axes.facecolor'] = 'skyblue' 
        ax.set_facecolor('skyblue')  # Set subplot background color
        plot_name += '_blue'
    else:
#         plt.rcParams['axes.facecolor'] = 'white'
        ax.set_facecolor('white')  # Set subplot background color
    if large_size:
        plot_name = '12X6_' + plot_name
    # Save the plot
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()
    
def plot_cost_comparison(data_dict, 
                         solver_name1, 
                         solver_name2, 
                         percNonUVals, 
                         VOTs, 
                         n_companies, 
                         factor = 10, 
                         blueBack = False, 
                         large_size=False,
                        fontsize = 20):

    if large_size:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))
    for percNonUVal_iter in percNonUVals:
        for (solver_name_temp, VOT, percNonUValTemp), n_company_dict in data_dict.items():
            if percNonUValTemp != percNonUVal_iter:
                continue
            if solver_name_temp != solver_name1 or VOT not in VOTs:
                continue
            if solver_name1 == "ADMM":
                solver_name_algo1 = "Algorithm 1"
            else:
                solver_name_algo1 = solver_name1
            if solver_name2 == "ADMM":
                solver_name_algo2 = "Algorithm 1"
            else:
                solver_name_algo2 = solver_name2
            percU = 100 - percNonUValTemp
            for n_company, (x_axis, cost_1) in n_company_dict.items():    
                if n_company not in n_companies:
                    continue
                # legend labels

                # x-axis
                x_axis_str = []
                for val in x_axis:
                    if val == 0:
                        x_axis_str.append('$0,\nNo Incentive')
                    else:
                        x_axis_str.append('$'+str(val))
                if large_size:
                    if n_company == 1:
                        label_str_n_company = '1 Organization'
                    elif n_company%factor == 0 or n_company <= 5:
                        label_str_n_company = str(n_company) + ' Organizations'
                    else:
                        label_str_n_company = 'Individual Drivers'
                    label_str = solver_name_algo1 + " vs " + solver_name_algo2 + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                else:
                    label_str_n_company = ""
#                     if len(solver_names) > 1 and len(VOTs) > 1:
#                         label_str = solver_name_algo1 + " vs " + solver_name_algo2 + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
#                     elif len(VOTs) > 1:
#                         label_str = str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
#                     elif len(solver_names) > 1:
#                         label_str = str(percU) + "%"
#                     else:
                    label_str = str(percU) + "%"
    #             print(label_str, x_axis, y_axis)
                cost_2 = data_dict[(solver_name2, VOT, percNonUValTemp)][n_company][1]
                assert len(cost_1)==len(cost_2), print("Number of costs are not equal")
                y_axis = [cost_1[i] - cost_2[i] for i in range(len(cost_1))]

                if large_size:
                    ax.plot(x_axis_str, y_axis, \
                        linewidth=8, marker='o', markersize=14, label=label_str)
                else:
                    ax.plot(x_axis_str, y_axis, \
                        linewidth=16, marker='o', markersize=28, label=label_str)
    if large_size:
        legend_title='Solver, VOT, Penetration Rate, # of Organization'
    else:
#         if len(solver_names) > 1 and len(VOTs) > 1:
#             legend_title='Solver, VOT, Penetration Rate'
#         elif len(VOTs) > 1:
#             legend_title='VOT, Penetration Rate'
#         elif len(solver_names) > 1:
#             legend_title='Solver, Penetration Rate'
#         else:
        legend_title='Penetration Rate'
    legend = plt.legend(title=legend_title, fontsize=fontsize-4,    # Smaller text
                                       markerscale=0.5,       # Smaller markers relative to plot
                                       handlelength=2,        # Shorter handles
                                       handleheight=1,        # Smaller handle height
                                       borderpad=0.5,         # Smaller border padding
                                       labelspacing=0.5,      # Less space between labels
                                       handletextpad=0.5)     # Less space between handle and text)
    legend.get_title().set_fontsize(fontsize-4)  # Set smaller font size for the legend title
    
    # plt.title(', '.join([solve_method, result_to_plot, 'tt_reduction_perc', args.region_]))
    solver_name_algo1 = "Algorithm 1" if solver_name1 == "ADMM" else solver_name1
    solver_name_algo2 = "Algorithm 1" if solver_name2 == "ADMM" else solver_name2
    ax.set_ylabel('Cost Difference\n(' + solver_name_algo1 + ' - ' + solver_name_algo2 + ')', fontsize = fontsize)
    ax.set_xlabel('Budget', fontsize = fontsize)
    # ax.set_ylim([0.0, 4000.0])
    # plt.ylim(0.0, 4000.0)
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    # Set font size for tick labels on both axes
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    # Plot name
    plot_name = 'costComparisonVOT' + \
                 '_VOT' + "_".join([str(x) for x in VOTs]) + \
                 '_solv1' +  solver_name1 + \
                 '_solv2' +  solver_name2 + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVals]) + \
                 '_nC' +  "_".join([str(x) for x in n_companies])    
    # Blue background for rebuttal
    if blueBack:
#         plt.rcParams['axes.facecolor'] = 'skyblue'
        ax.set_facecolor('skyblue')  # Set subplot background color
        plot_name += '_blue'
    else:
#         plt.rcParams['axes.facecolor'] = 'white'
        ax.set_facecolor('white')  # Set subplot background color
            
    if large_size:
        plot_name = '12X6_' + plot_name
    # Save the plot
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()
    
def plot_cost_new(data_dict, 
                  solver_names, 
                  percNonUVals, 
                  VOTs, 
                  n_companies, 
                  factor = 10, 
                  blueBack = False, 
                  large_size=False,
                fontsize = 20):

    if large_size:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))
        
    for percNonUVal_iter in percNonUVals:
        for (solver_name_temp, VOT, percNonUValTemp), n_company_dict in data_dict.items():
            if percNonUValTemp != percNonUVal_iter or solver_name_temp not in solver_names or VOT not in VOTs:
                continue
            if solver_name_temp == "ADMM":
                solver_name_algo = "Algorithm 1"
            else:
                solver_name_algo = solver_name_temp
            percU = 100 - percNonUValTemp
            for n_company, (x_axis, y_axis) in n_company_dict.items():    
                if n_company not in n_companies:
                    continue
                # legend labels

                # x-axis
                x_axis_str = []
                for val in x_axis:
                    if val == 0:
                        x_axis_str.append('$0,\nNo Incentive')
                    else:
                        x_axis_str.append('$'+str(val))
                if large_size:
                    if n_company == 1:
                        label_str_n_company = '1 Organization'
                    elif n_company%factor == 0 or n_company <= 5:
                        label_str_n_company = str(n_company) + ' Organizations'
                    else:
                        label_str_n_company = 'Individual Drivers'
                    label_str = solver_name_algo + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                else:
                    label_str_n_company = ""
                    if len(solver_names) > 1 and len(VOTs) > 1:
                        label_str = solver_name_algo + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                    elif len(VOTs) > 1:
                        label_str = str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                    elif len(solver_names) > 1:
                        label_str = str(percU) + "%" + ", " + solver_name_algo + ", " + label_str_n_company
                    else:
                        label_str = str(percU) + "%"
    #             print(label_str, x_axis, y_axis)
                y_axis_plot = y_axis
                if large_size:
                    ax.plot(x_axis_str, y_axis_plot, \
                        linewidth=8, marker='o', markersize=14, label=label_str)
                else:
                    ax.plot(x_axis_str, y_axis_plot, \
                        linewidth=16, marker='o', markersize=28, label=label_str)


    if large_size:
        legend_title='Solver, VOT, Penetration Rate, # of Organization'
    else:
        if len(solver_names) > 1 and len(VOTs) > 1:
            legend_title='Solver, VOT, Penetration Rate'
        elif len(VOTs) > 1:
            legend_title='VOT, Penetration Rate'
        elif len(solver_names) > 1:
            legend_title='Penetration Rate, Solver'
        else:
            legend_title='Penetration Rate'
        
    legend = plt.legend(title=legend_title, fontsize=fontsize-4,    # Smaller text
                                       markerscale=0.5,       # Smaller markers relative to plot
                                       handlelength=2,        # Shorter handles
                                       handleheight=1,        # Smaller handle height
                                       borderpad=0.5,         # Smaller border padding
                                       labelspacing=0.5,      # Less space between labels
                                       handletextpad=0.5)     # Less space between handle and text)
    legend.get_title().set_fontsize(fontsize-4)  # Set smaller font size for the legend title
    
    # plt.title(', '.join([solve_method, result_to_plot, 'tt_reduction_perc', args.region_]))
    ax.set_ylabel('Total Cost', fontsize = fontsize)
    ax.set_xlabel('Budget', fontsize = fontsize)
    if max(y_axis_plot)<4000:
        ax.set_ylim([0.0, 4000.0])
#     ax.set_ylim([0.0, max(math.ceil(max(y_axis_plot)/100)*100,4000.0)])
    # plt.ylim(0.0, 4000.0)
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    # Set font size for tick labels on both axes
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    # Plot name
    plot_name = 'costVOT' + \
                 '_VOT' + "_".join([str(x) for x in VOTs]) + \
                 '_solv' +  "_".join([str(x) for x in solver_names]) + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVals]) + \
                 '_nC' +  "_".join([str(x) for x in n_companies])    
    # Blue background for rebuttal
    if blueBack:
#         plt.rcParams['axes.facecolor'] = 'skyblue' 
        ax.set_facecolor('skyblue')  # Set subplot background color
        plot_name += '_blue'
    else:
#         plt.rcParams['axes.facecolor'] = 'white'
        ax.set_facecolor('white')  # Set subplot background color
    if large_size:
        plot_name = '12X6_' + plot_name
    # Save the plot
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    
def plot_cost_per_driver_new(data_dict, 
                             solver_names, 
                             percNonUVals, 
                             VOTs, 
                             n_companies, 
                             factor = 10, 
                             blueBack = False, 
                             large_size=False,
                            fontsize = 20):

    if large_size:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))
    for (solver_name_temp, VOT, percNonUValTemp), n_company_dict in data_dict.items():
        if percNonUValTemp not in percNonUVals or solver_name_temp not in solver_names or VOT not in VOTs:
            continue
        if solver_name_temp == "ADMM":
            solver_name_algo = "Algorithm 1"
        else:
            solver_name_algo = solver_name_temp
        percU = 100 - percNonUValTemp
        for n_company, (x_axis, y_axis) in n_company_dict.items():    
            if n_company not in n_companies:
                continue
            # legend labels
            if large_size:
                if n_company == 1:
                    label_str_n_company = '1 Organization'
                elif n_company%factor == 0 or n_company <= 5:
                    label_str_n_company = str(n_company) + ' Organizations'
                else:
                    label_str_n_company = 'Individual Drivers'
                label_str = solver_name_algo + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
            else:
                label_str_n_company = ""
                if len(solver_names) > 1 and len(VOTs) > 1:
                    label_str = solver_name_algo + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                elif len(VOTs) > 1:
                    label_str = str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                elif len(solver_names) > 1:
                    label_str = solver_name_algo + ", " + str(percU) + "%" + ", " + label_str_n_company
                else:
                    label_str = str(percU) + "%"

            # x-axis
            x_axis_str = []
            for val in x_axis:
                if val == 0:
                    x_axis_str.append('$0,\nNo Incentive')
                else:
                    x_axis_str.append('$'+str(val))
#             print(label_str, x_axis, y_axis)

            if large_size:
                ax.plot(x_axis_str, y_axis, \
                    linewidth=8, marker='o', markersize=14, label=label_str)
            else:
                ax.plot(x_axis_str, y_axis, \
                    linewidth=16, marker='o', markersize=28, label=label_str)
            
    # Show legends
    if large_size:
        legend_title='Solver, VOT, Penetration Rate, # of Organization'
    else:
        if len(solver_names) > 1 and len(VOTs) > 1:
            legend_title='Solver, VOT, Penetration Rate'
        elif len(VOTs) > 1:
            legend_title='VOT, Penetration Rate'
        elif len(solver_names) > 1:
            legend_title='Solver, Penetration Rate'
        else:
            legend_title='Penetration Rate'
    legend = ax.legend(title=legend_title, 
                       fontsize=fontsize-4,           # Smaller text
                       markerscale=0.5,       # Smaller markers relative to plot
                       handlelength=2,        # Shorter handles
                       handleheight=1,        # Smaller handle height
                       borderpad=0.5,         # Smaller border padding
                       labelspacing=0.5,      # Less space between labels
                       handletextpad=0.5)     # Less space between handle and text)
    legend.get_title().set_fontsize(fontsize-4)
        
#     ax2.legend(loc='upper right')
    
    # plt.title(', '.join([solve_method, result_to_plot, 'tt_reduction_perc', args.region_]))
    ax.set_ylabel('Cost Per Deviated Driver', fontsize = fontsize)
    ax.set_xlabel('Budget', fontsize = fontsize)
    # ax.set_ylim([0.0, 4000.0])
    # plt.ylim(0.0, 4000.0)
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    # Set font size for tick labels on both axes
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    # Plot name
    plot_name = 'costPerDeviatedDriver' + \
                 '_VOT' + "_".join([str(x) for x in VOTs]) + \
                 '_solv' +  "_".join([str(x) for x in solver_names]) + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVals]) + \
                 '_nC' +  "_".join([str(x) for x in n_companies])    
    
    # Blue background for rebuttal
    if blueBack:
#         plt.rcParams['axes.facecolor'] = 'skyblue' 
        ax.set_facecolor('skyblue')  # Set subplot background color
        plot_name += '_blue'
    else:
#         plt.rcParams['axes.facecolor'] = 'white'
        ax.set_facecolor('white')  # Set subplot background color
    
    if large_size:
        plot_name = '12X6_' + plot_name
    # Save the plot
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()
    
    
def plot_cost_per_driver_with_number_new(data_dict, 
                                         data_dict2, 
                                         solver_names, 
                                         percNonUVals, 
                                         VOTs, 
                                         n_companies, 
                                         factor = 10, 
                                         blueBack = False, 
                                         large_size=False,
                                        fontsize = 20):

    if large_size:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))
    for (solver_name_temp, VOT, percNonUValTemp), n_company_dict in data_dict.items():
        if percNonUValTemp not in percNonUVals or solver_name_temp not in solver_names or VOT not in VOTs:
            continue
        if solver_name_temp == "ADMM":
            solver_name_algo = "Algorithm 1"
        else:
            solver_name_algo = solver_name_temp
        percU = 100 - percNonUValTemp
        for n_company, (x_axis, y_axis) in n_company_dict.items():    
            if n_company not in n_companies:
                continue
            # legend labels
            if large_size:
                if n_company == 1:
                    label_str_n_company = '1 Organization'
                elif n_company%factor == 0 or n_company <= 5:
                    label_str_n_company = str(n_company) + ' Organizations'
                else:
                    label_str_n_company = 'Individual Drivers'
                label_str = solver_name_algo + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
            else:
                label_str_n_company = ""
                if len(solver_names) > 1 and len(VOTs) > 1:
                    label_str = solver_name_algo + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                elif len(VOTs) > 1:
                    label_str = str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                elif len(solver_names) > 1:
                    label_str = solver_name_algo + ", " + str(percU) + "%" + ", " + label_str_n_company
                else:
                    label_str = str(percU) + "%"

            # Plot line plot on secondary axis
            x_axis_str2 = []
            for val in data_dict2[(solver_name_temp, VOT, percNonUValTemp)][0]:
                if val == 0:
                    x_axis_str2.append('$0,\nNo Incentive')
                else:
                    x_axis_str2.append('$'+str(val))
            y2 = data_dict2[(solver_name_temp, VOT, percNonUValTemp)][1]
            ax.bar(x_axis_str2, y2, label=label_str)
            ax.set_ylabel('Number of Deviated Drivers')

            # Create secondary axis
            ax2 = ax.twinx()
            # x-axis
            x_axis_str = []
            for val in x_axis:
                if val == 0:
                    x_axis_str.append('$0,\nNo Incentive')
                else:
                    x_axis_str.append('$'+str(val))
#             print(label_str, x_axis, y_axis)

            if large_size:
                ax2.plot(x_axis_str, y_axis, \
                    linewidth=8, marker='o', markersize=14, label=label_str)
            else:
                ax2.plot(x_axis_str, y_axis, \
                    linewidth=16, marker='o', markersize=28, label=label_str)
            ax2.set_ylabel('Cost Per Deviated Driver')
    # Show legends
    if large_size:
        legend_title='Solver, VOT, Penetration Rate, # of Organization'
    else:
        if len(solver_names) > 1 and len(VOTs) > 1:
            legend_title='Solver, VOT, Penetration Rate'
        elif len(VOTs) > 1:
            legend_title='VOT, Penetration Rate'
        elif len(solver_names) > 1:
            legend_title='Solver, Penetration Rate'
        else:
            legend_title='Penetration Rate'
            
    legend = ax.legend(title=legend_title, 
                       fontsize=fontsize-4,           # Smaller text
                       markerscale=0.5,       # Smaller markers relative to plot
                       handlelength=2,        # Shorter handles
                       handleheight=1,        # Smaller handle height
                       borderpad=0.5,         # Smaller border padding
                       labelspacing=0.5,      # Less space between labels
                       handletextpad=0.5)     # Less space between handle and text)
    legend.get_title().set_fontsize(fontsize-4)
        
#     ax2.legend(loc='upper right')
    
    # plt.title(', '.join([solve_method, result_to_plot, 'tt_reduction_perc', args.region_]))
    
    ax.set_xlabel('Budget', fontsize = fontsize)
    # ax.set_ylim([0.0, 4000.0])
    # plt.ylim(0.0, 4000.0)
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax2.yaxis.set_major_formatter(tick)
    # Set font size for tick labels on both axes
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    # Plot name
    plot_name = 'costPerDeviatedDriverNumber' + \
                 '_VOT' + "_".join([str(x) for x in VOTs]) + \
                 '_solv' +  "_".join([str(x) for x in solver_names]) + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVals]) + \
                 '_nC' +  "_".join([str(x) for x in n_companies])    
    
    # Blue background for rebuttal
    if blueBack:
#         plt.rcParams['axes.facecolor'] = 'skyblue' 
        ax.set_facecolor('skyblue')  # Set subplot background color
        plot_name += '_blue'
    else:
#         plt.rcParams['axes.facecolor'] = 'white'
        ax.set_facecolor('white')  # Set subplot background color
    
    if large_size:
        plot_name = '12X6_' + plot_name
    # Save the plot
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()
    
def plot_cost_tt_reduction_new(data_dict, 
                               solver_names, 
                               percNonUVals, 
                               VOTs, 
                               factor = 10, 
                               blueBack = False, 
                               large_size=False,
                                fontsize = 20, 
                              skip_n_company = [2,3,5]):

    if large_size:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))
    for (solver_name_temp, VOT, percNonUValTemp), n_company_dict in data_dict.items():
        if percNonUValTemp not in percNonUVals or solver_name_temp not in solver_names or VOT not in VOTs:
            continue
        if solver_name_temp == "ADMM":
            solver_name_algo = "Algorithm 1"
        percU = 100 - percNonUValTemp
        for n_company, (x_axis, y_axis) in n_company_dict.items():    
            if n_company in skip_n_company:
                continue
            # legend labels
            if n_company == 1:
                label_str_n_company = '1 Organization'
            elif n_company%factor == 0 or n_company <= 5:
                label_str_n_company = str(n_company) + ' Organizations'
            else:
                label_str_n_company = 'Individual Drivers'
            
            if large_size:
                label_str = solver_name_temp + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
            else:
                if len(solver_names) > 1 and len(VOTs) > 1 and len(percNonUVals) > 1:
                    label_str = solver_name_temp + ", " + str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                elif len(VOTs) > 1:
                    label_str = str(VOT) + ", " + str(percU) + "%" + ", " + label_str_n_company
                elif len(solver_names) > 1:
                    label_str = solver_name_temp + ", " + str(percU) + "%" + ", " + label_str_n_company
                else:
                    label_str = label_str_n_company
#             print(label_str, x_axis, y_axis)

            if large_size:
                ax.plot(x_axis, y_axis, \
                    linewidth=8, marker='o', markersize=14, label=label_str)
            else:
                ax.plot(x_axis, y_axis, \
                    linewidth=16, marker='o', markersize=28, label=label_str)

    if large_size:
        legend_title='Solver, VOT, Penetration Rate, No. of Orgs.'
    else:
        if len(solver_names) > 1 and len(VOTs) > 1:
            legend_title='Solver, VOT, Penetration Rate, No. of Orgs.'
        elif len(VOTs) > 1:
            legend_title='VOT, Penetration Rate, No. of Orgs.'
        elif len(solver_names) > 1:
            legend_title='Solver, Penetration Rate, No. of Orgs.'
        else:
            legend_title='No. of Orgs.'
            
    legend = ax.legend(title=legend_title, 
                       fontsize=fontsize-4,           # Smaller text
                       markerscale=0.5,       # Smaller markers relative to plot
                       handlelength=2,        # Shorter handles
                       handleheight=1,        # Smaller handle height
                       borderpad=0.5,         # Smaller border padding
                       labelspacing=0.5,      # Less space between labels
                       handletextpad=0.5)     # Less space between handle and text)
    legend.get_title().set_fontsize(fontsize-4)
        
    # plt.legend(title='Penetration Rate (%)').remove()
    # plt.title(', '.join([solve_method, result_to_plot, 'tt_reduction_perc', args.region_]))
    ax.set_ylabel('Travel Time Reduction', fontsize = fontsize)
    ax.set_xlabel('Cost', fontsize = fontsize)
    # ax.set_ylim([0.0, 4000.0])
    # plt.ylim(0.0, 4000.0)
    fmt = '{x:,.2f}%'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    fmt2 = '${x:,.0f}'
    tick2 = mtick.StrMethodFormatter(fmt2)
    ax.xaxis.set_major_formatter(tick2)
    # Set font size for tick labels on both axes
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()
    

    # Plot name
    plot_name = 'costTTReduction' + \
                 '_VOT' + "_".join([str(x) for x in VOTs]) + \
                 '_solv' +  "_".join([str(x) for x in solver_names]) + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVals])
    
    # Blue background for rebuttal
    if blueBack:
#         plt.rcParams['axes.facecolor'] = 'skyblue' 
        ax.set_facecolor('skyblue')  # Set subplot background color
        plot_name += '_blue'
    else:
#         plt.rcParams['axes.facecolor'] = 'white'
        ax.set_facecolor('white')  # Set subplot background color
    
    if large_size:
        plot_name = '12X6_' + plot_name
    # Save the plot
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)
    plt.show()

    
def plot_execution_time_comparison(data_dict, percNonUVal_list, blueBack, large_size=False, fontsize=28):
    if large_size:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))

    # Data
    times_algo1 = [int(round(x)) for x in data_dict['Gurobi']]
    times_algo2 = [int(round(x)) for x in data_dict['Mosek']]

    x = np.arange(len(percNonUVal_list))  # the label locations
    width = 0.35  # the width of the bars

    # Plotting data
    rects1 = ax.bar(x - width/2, times_algo1, width, label='Gurobi')
    rects2 = ax.bar(x + width/2, times_algo2, width, label='MOSEK')

    # Adding text labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}X',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Penetration Rate')
    ax.set_ylabel('Relative Execution Time of\nAlgorithm 1 Compared to Solvers')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{100-value:.0f}%" for value in percNonUVal_list])

    legend = plt.legend(title='Solver', 
                        loc='upper left', 
                        fontsize=fontsize-4,    # Smaller text
                        markerscale=0.5,       # Smaller markers relative to plot
                        handlelength=2,        # Shorter handles
                        handleheight=1,        # Smaller handle height
                        borderpad=0.5,         # Smaller border padding
                        labelspacing=0.5,      # Less space between labels
                        handletextpad=0.5)     # Less space between handle and text)
    legend.get_title().set_fontsize(fontsize-4)  # Set smaller font size for the legend title

    # Set y-axis limit higher to accommodate labels
    ax.set_ylim(0, max(times_algo1 + times_algo2) * 1.2)  # 20% higher than the tallest bar

    ax.tick_params(axis='both', labelsize=fontsize)

    # Use tight_layout to automatically adjust subplot params
    plt.tight_layout()        

    # Plot name
    plot_name = 'execTimeComparison' + \
                 '_percNonU' +  "_".join([str(x) for x in percNonUVal_list])

    # Blue background for rebuttal
    if blueBack:
    #         plt.rcParams['axes.facecolor'] = 'skyblue' 
        ax.set_facecolor('skyblue')  # Set subplot background color
        plot_name += '_blue'
    else:
        ax.set_facecolor('white')  # Set subplot background color
    if large_size:
        plot_name = '12X6_' + plot_name
    # Save the plot
    plt.savefig(os.path.join('..', 'plots', plot_name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join('..', 'plots', plot_name + '.pdf'), bbox_inches='tight', pad_inches=0)

    # Show plot
    plt.show()