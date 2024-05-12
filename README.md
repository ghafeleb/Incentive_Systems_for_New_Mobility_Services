# Incentive Systems for New Mobility Services
![organization-level](https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/incentiveOfferingPlatforms.png?raw=true)
Traffic congestion has become an inevitable challenge in large cities due to the increase in population and expansion of urban areas. Studies have introduced a variety of approaches to mitigate traffic issues, encompassing methods from expanding road infrastructure to employing demand management. Congestion pricing and incentive schemes are extensively studied for traffic control in traditional networks where each driver is a network "player". In this setup, drivers' "selfish" behavior hinders the network from reaching a socially optimal state. In future mobility services, on the other hand, a large portion of drivers/vehicles may be controlled by a small number of companies/organizations. In such a system,  offering incentives to organizations can potentially be much more effective in reducing traffic congestion rather than offering incentives directly to drivers. This paper studies the problem of offering incentives to organizations to change the behavior of their individual drivers (or individuals relying on the organization’s services). We developed a model where incentives are offered to each organization based on the aggregated travel time loss across all drivers in that organization. Such an incentive offering mechanism requires solving a large-scale optimization problem to minimize the system-level travel time. We propose an efficient algorithm for solving this optimization problem. Numerous experiments on Los Angeles County traffic data reveal the ability of our method to reduce system-level travel time by up to 6.9%. Moreover, our experiments demonstrate that incentivizing organizations can be up to 8 times more efficient than incentivizing individual drivers in terms of incentivization monetary cost.

# Data
We evaluate our incentive scheme's effectiveness using Los Angeles area data. The presence of multiple routes between most origin-destination (OD) pairs makes the Los Angeles area particularly suitable for our assessment. We use the data collected by the Archived Data Management System (ADMS), a comprehensive transportation dataset compilation by University of Southern California researchers. First, we extract sensor details, including their locations. We extract the speed and volume data of selected sensors. Nodes for the network graph are chosen from on-ramps and highway intersections. Connecting link data is derived from in-between sensors. Node distances are determined via Google Maps API. The data preparation workflow is as follows:

<p align="center">
<img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/data_preparation_workflow.png" width="400" />
</p>

Our network encompasses highways around Downtown Los Angeles and has 12 nodes, 32 links, 288.1 miles of road, 144 OD pairs, and 270 paths between OD pairs.

<p align="center">
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/region_y3_new.PNG" width="600" />
<p align="center">
  
The Los Angeles County data cannot be shared with the public. Synthetic data is provided to check the code. The example network is a graph with 4 nodes and one OD pair. There are two paths between ODs, and each path includes two edges.

# Demo
To test a demo of the incentivization platform, run 
```
cd scripts
python runDemo.py
```
This demo creates the synthetic graph. Next, it incentivizes the user drivers of the system. Finally, it computes the incentivization cost.

# Numerical Experiments
The number of organizations in the system can alter the total travel time and cost. Following figures illustrate the percentage decrease of travel time and total cost when there are different number of organizations in the system. As an extreme case, we also include the case that each organization contains one driver (i.e., we incentivize individuals rather than organizations). In the following figures, we observe a larger cost for reducing the same amount of travel time decrease when there are more organizations in the system. The intuitive reason behind this observation is as follows. For each organization, after incentivization, some drivers lose time, and some gain travel time. At the organizational level, the time changes of drivers can cancel each other out, and hence we may not need to compensate the organization significantly. When the number of drivers per organization decreases, the canceling effect becomes weaker, and the incentivization costs more. This also explains why incentivizing organizations is much more cost-efficient than incentivizing individual drivers.

<p float="left">
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/cost_tt_reduction_scenario1_VOT157.png" width="400" />
  <img src="https://github.com/ghafeleb/Incentive_Systems_for_New_Mobility_Services/blob/main/images/cost_tt_reduction_scenario2_VOT157.png" width="400" />
</p> 
