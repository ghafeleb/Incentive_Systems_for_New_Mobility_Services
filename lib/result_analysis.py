import itertools

def get_max_tt_reduction(budget_tt_reduction_plot_dict, solver_name):
    max_tt_reduction = -1
    for (solver, VOT, percNonUVal), (budgets, tt_reductions) in budget_tt_reduction_plot_dict.items():
        if solver == solver_name:
            max_tt_reduction = max(max_tt_reduction, max(tt_reductions))

    return max_tt_reduction

def get_max_one_org_vs_individuals(cost_tt_reduction_plot_dict, solver_name, ):
    max_one_org_vs_individuals = -1
    for (solver, VOT, percNonUVal), org_cost_tt_reductions in cost_tt_reduction_plot_dict.items():
        if solver != solver_name:
            continue
        n_orgs = list(cost_tt_reduction_plot_dict[(solver, VOT, percNonUVal)].keys())
        cost_one_org = org_cost_tt_reductions[min(n_orgs)][0]
        cost_individuals = org_cost_tt_reductions[max(n_orgs)][0]
        assert len(cost_one_org) == len(cost_individuals), print('Not the same number of costs recorded!')
        one_org_vs_individuals = [round(cost_individuals[idx+1]/cost_one_org[idx+1], 2) for idx in range(len(cost_one_org)-1)] # Skip 0 cost
        print(f"\nsolver={solver}, VOT={VOT}, percNonUVal={percNonUVal}")
#         print("Cost of one company incentivization at different budgets: ", cost_one_org)
#         print("Cost of individual incentivization at different budgets: ", cost_individuals)
        print(f"Cost of 1 org. vs individuals at maximum budget: ", one_org_vs_individuals[-1])
#         max_one_org_vs_individuals = max(max_one_org_vs_individuals, max(one_org_vs_individuals))
        max_one_org_vs_individuals = max(max_one_org_vs_individuals, one_org_vs_individuals[-1])
    return max_one_org_vs_individuals


def get_comparison_VOT_tt_decrease(solver_solution_dict, solver_name1, solver_name2, VOT1, VOT2, budgets, percNonUVals):
    print(f"Comparison of travel time decrease of solver={solver_name1} at VOT={VOT1} and solver={solver_name2} at VOT={VOT2}")
    max_VOT_tt_decrease = -float('inf')
    for budget, percNonUVal in itertools.product(budgets, percNonUVals):
        larger_VOT_tt_decrease = solver_solution_dict[(solver_name1, VOT1, percNonUVal, budget)]['tt_decrease_perc']
        smaller_VOT_tt_decrease = solver_solution_dict[(solver_name2, VOT2, percNonUVal, budget)]['tt_decrease_perc']
        diff_tt_decrease =  smaller_VOT_tt_decrease - larger_VOT_tt_decrease
        print(f"\nPenetration rate={100-percNonUVal}%, Budget=${budget}\n>> Difference at travel time decrease (solver1@VOT1 - solver2@VOT2): {round(diff_tt_decrease, 2)}%")
        max_VOT_tt_decrease = max(max_VOT_tt_decrease, round(diff_tt_decrease, 2))
    return max_VOT_tt_decrease
