@echo off
echo "Run DataCreator1"
python2 create_graph_py2.py
python3 create_graph_py3.py
python3 DataCreator1.py --config_filename=data/YAML/region_toy_DataCreator1.yaml
echo "Running Det vCVX..."
SET /A iter = 5
SET /A percNonUVal = 50
SET /A n_time = 204
SET /A n_time_inc_start = 13
SET /A n_time_inc_end = 24
python2 HistoricalData.py --config_filename=data/YAML/region_toy_HistoricalData.yaml
for /l %%n in (0,1,%iter%) do ( 
call echo %%n
call matlab -r -wait -nodesktop -nojvm "try;DataLoader_Det(%%n,%percNonUVal%,%n_time%,%n_time_inc_start%,%n_time_inc_end%);catch;end;quit";
python2 runDet.py --config_filename=data/YAML/region_toy_runDet.yaml --sD=2 --nC=2 --f=0_0_0_100_0 --percC=50_50 --percNonU=%percNonUVal% --iterRun=%%n --initEta=-1
python3 DataCreator2.py --config_filename=data/YAML/region_toy_DataCreator2.yaml --DetStoch=Det --sD=2 --nC=2 --f=0_0_0_100_0 --percC=50_50 --percNonU=%percNonUVal% --iterRun=%%n --initEta=-1
call echo "Next Iter:"
)
matlab -r -wait "try;initEta_prestep(%iter%,%percNonUVal%,%n_time%,%n_time_inc_start%,%n_time_inc_end%);catch;end;quit";
python2 runDet.py --config_filename=data/YAML/region_toy_runDet.yaml --sD=2 --nC=2 --f=0_0_0_100_0 --percC=50_50 --percNonU=%percNonUVal% --iterRun=%iter% --initEta=1
python3 DataCreator2.py --config_filename=data/YAML/region_toy_DataCreator2.yaml --DetStoch=Det --sD=2 --nC=2 --f=0_0_0_100_0 --percC=50_50 --percNonU=%percNonUVal% --iterRun=%iter% --initEta=1
SET /A budget = 10000
SET /A MaxIterADMM = 100
matlab -r -wait "try;ADMM(%iter%,%percNonUVal%,%budget%,%MaxIterADMM%,%n_time%,%n_time_inc_start%,%n_time_inc_end%);catch;end;quit";
echo "Compute Real Cost"
call matlab -r -wait -nodesktop -nojvm "try;getSV_realCost(%iter%,%percNonUVal%,%budget%,1,%MaxIterADMM%,%n_time%,%n_time_inc_start%,%n_time_inc_end%);catch;end;quit";
python2 run_realCost.py --sA=2 --sD=2 --nC=2 --f=0_0_0_100_0 --percC=50_50 --percNonU=%percNonUVal% --iterRun=%iter% --b=%budget% --nTIS=%n_time_inc_start% --nTIE=%n_time_inc_end%  --it=%MaxIterADMM% --nC=1
call matlab -r -wait -nodesktop -nojvm "try;compareCosts_realCost_initAll2(%iter%,%percNonUVal%,%budget%,1,%MaxIterADMM%,%n_time%,%n_time_inc_start%,%n_time_inc_end%);catch;end";
::call matlab -r -wait -nodesktop -nojvm "try;compareCosts_realCost_initAll2(%iter%,%percNonUVal%,%budget%,10,%MaxIterADMM%,%n_time%,%n_time_inc_start%,%n_time_inc_end%);catch;end";
::call matlab -r -wait -nodesktop -nojvm "try;compareCosts_realCost_initAll2(%iter%,%percNonUVal%,%budget%,100,%MaxIterADMM%,%n_time%,%n_time_inc_start%,%n_time_inc_end%);catch;end";
::call matlab -r -wait -nodesktop -nojvm "try;compareCosts_realCost_initAll2(%iter%,%percNonUVal%,%budget%,10000,%MaxIterADMM%,%n_time%,%n_time_inc_start%,%n_time_inc_end%);catch;end";
::cmd /k
PAUSE