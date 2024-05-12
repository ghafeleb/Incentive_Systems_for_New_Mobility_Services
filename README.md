# Incentive Systems for New Mobility Services
<p align="center">
<img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/incentiveOfferingPlatforms.png" width="1200" alt="Traditional vs. Organization Incentivization"/>
  <br>
  <em>(a) Traditional platforms for offering incentives: incentives are offered to individual drivers in the system. (b) Presented platform for offering incentives: incentives are offered to new mobility services to change their drivers' behavior.</em>
</p>
  
Traffic congestion has become an inevitable challenge in large cities due to the increase in population and expansion of urban areas. Studies have introduced a variety of approaches to mitigate traffic issues, encompassing methods from expanding road infrastructure to employing demand management. Congestion pricing and incentive schemes are extensively studied for traffic control in traditional networks where each driver is a network "player". In this setup, drivers' "selfish" behavior hinders the network from reaching a socially optimal state. In future mobility services, on the other hand, a large portion of drivers/vehicles may be controlled by a small number of companies/organizations. In such a system,  offering incentives to organizations can potentially be much more effective in reducing traffic congestion rather than offering incentives directly to drivers. This paper studies the problem of offering incentives to organizations to change the behavior of their individual drivers (or individuals relying on the organization’s services). We developed a model where incentives are offered to each organization based on the aggregated travel time loss across all drivers in that organization. Such an incentive offering mechanism requires solving a large-scale optimization problem to minimize the system-level travel time. We propose an efficient algorithm for solving this optimization problem. Numerous experiments on Los Angeles County traffic data reveal the ability of our method to reduce system-level travel time by up to 6.9%. Moreover, our experiments demonstrate that incentivizing organizations can be up to 8 times more efficient than incentivizing individual drivers in terms of incentivization monetary cost.

# Dependencies
The required packages must be installed via Anaconda or pip before running the codes.

Download and install **Python 2.x version** from [Python 2.x Version](https://www.python.org/downloads/). You can install the required packages via pip using the following command:
```
python2 -m pip install -r requirements.txt
```
Download and install **Python 3.x version** from [Python 3.x Version](https://www.python.org/downloads/).  You can install the required packages via pip using the following command:
```
python3 -m pip install -r requirements.txt
```


# Data
We evaluate our incentive scheme's effectiveness using Los Angeles area data. The presence of multiple routes between most origin-destination (OD) pairs makes the Los Angeles area particularly suitable for our assessment. We use the data collected by the Archived Data Management System (ADMS), a comprehensive transportation dataset compilation by University of Southern California researchers. First, we extract sensor details, including their locations. We extract the speed and volume data of selected sensors. Nodes for the network graph are chosen from on-ramps and highway intersections. Connecting link data is derived from in-between sensors. Node distances are determined via Google Maps API. The data preparation workflow is as follows:

<p align="center">
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/data_preparation_workflow.png" width="400" alt="Data Preparation Workflow"/>
  <br>
  <em>Data preparation workflow: First, traffic data and sensors' location data are received from ADMS Server. Next, sensors' location data is processed to compute sensor distances. Finally, sensor distances and traffic data are combined to create the graph network data.</em>
</p>

Our network encompasses highways around Downtown Los Angeles and has 12 nodes, 32 links, 288.1 miles of road, 144 OD pairs, and 270 paths between OD pairs.

<p align="center">
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/region_y3_new.PNG" width="600" />
  <br>
  <em>Studied region and the highway sensors inside the region. This region encompasses several areas notorious for high traffic congestion, particularly Downtown Los Angeles.</em>
<p align="center">
  
The Los Angeles County data cannot be shared with the public, but synthetic data is provided in the demo to check the code. 

# Demo
You can test the incentivization on a synthetic graph network by running the following command:
```
cd scripts
python runDemo.py --region_ "region_toy" --config_filename "../data/YAML/region_toy.yaml" --n_iter_UE 5 --seed_data 2 --seed_solver 2 --budget 10000 --VOT 2.63 --solver_name "ADMM" --MIPGap 0.01 --n_iter_ADMM 100 --step_size_UE 0.01 --percNonUVal 50 --nPath 2 --n_time 204 --n_time_inc_start 13 --n_time_inc_end 24 --rho 20 --n_companies_ADMM 1
```
First, the demo creates the synthetic data as a demo example. The demo graph includes 4 nodes (O, A, B, and D) and one Origin-Destination (OD) pair: node O as the origin and node D as the destination. There are 4 roads/edges (x, y, z, and w) 2 routes between the OD (route 1: x-y, route 2: z-w). After creating the data, the demo incentivizes the user drivers of the system. Finally, it computes the incentivization cost.

<p align="center">
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/synthetic_data.png" width="300" alt="Demo Graph"/>
  <br>
  <em>Demo graph.</em>
<p align="center">
  
  
# Numerical Experiments
## Incentivization Cost Analysis
The number of organizations in the system can alter the total travel time and cost. The following figures illustrate the percentage decrease in travel time and total cost when there are different numbers of organizations in the system. As an extreme case, we also include the case that each organization contains one driver (i.e., we incentivize individuals rather than organizations). In the following figures, we observe a larger cost for reducing the same amount of travel time decrease when there are more organizations in the system. The intuitive reason behind this observation is as follows. For each organization, some drivers lose time after incentivization, and some gain travel time. At the organizational level, the time changes of drivers can cancel each other out, and hence we may not need to compensate the organization significantly. When the number of drivers per organization decreases, the canceling effect becomes weaker, and the incentivization costs more. This also explains why incentivizing organizations is much more cost-efficient than incentivizing individual drivers.

<p float="left">
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/cost_tt_reduction_scenario1_VOT157.png" width="400" />
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/cost_tt_reduction_scenario2_VOT157.png" width="400" />
</p> 

## Algorithm Performance Analysis
We compare our presented algorithm against Gurobi and MOSEK as state-of-the-art commercial solvers. Our algorithm 
* Achieves speeds up to 12 times faster than Gurobi and 120 times faster than MOSEK
* Closely mirrors the performance of Gurobi and Mosek
* Saves up to $5000 in incentivization cost
<p align="center">
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/execTimeComparison_percNonU95_90_85_80.png" width="500" alt="Execution Time Comparison"/>
  <br>
  <em> The presented algorithm significantly outperforms Gurobi and MOSEK in execution time.</em>
<p align="center">
  
