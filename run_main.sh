#!/bin/bash

# 3. Dataset Generation
python -u pipeline/b_fNIRS_database_generate.py --load_dir /data/eeggroup/public_dataset/ --save_dir /data/eeggroup/CL_database/ > ~/CL_logs/database_fNIRS_2.log
python -u pipeline/b_Sleep_database_generate.py --load_dir /data/eeggroup/public_dataset/ --save_dir /data/eeggroup/CL_database/ > ~/CL_logs/database_Sleep.log
python -u pipeline/b_HHAR_database_generate.py --load_dir /data/eeggroup/public_dataset/ --save_dir /data/eeggroup/CL_database/ > ~/CL_logs/database_HHAR.log

# 4. Run the model
for r in 0 0.2 0.4
do
  for i in {1..12}
	do
	  python -u a_train.py --database_save_dir /data/eeggroup/CL_database/ --data_name fNIRS_2 --noise_ratio $r --exp_id $i --gpu_id 0 --path_checkpoint /data/eeggroup/CL_result/ > ~/CL_logs/cl_train_fNIRS_2_r$r.$i.log
	  python -u b_test.py --database_save_dir /data/eeggroup/CL_database/ --data_name fNIRS_2 --noise_ratio $r --exp_id $i --gpu_id 0 --load_path /data/eeggroup/CL_result/ > ~/CL_logs/cl_test_fNIRS_2_r$r.$i.log
	  wait
	done
	wait

	python -u b_test.py --data_name fNIRS_2 --noise_ratio $r --load_path /data/eeggroup/CL_result/ --summary True
	wait
done
wait


for r in 0 0.2 0.4
do
  for i in {1..6}
  do
    python -u a_train.py --database_save_dir /data/eeggroup/CL_database/ --data_name Sleep --noise_ratio $r --exp_id $i --gpu_id 0 --path_checkpoint /data/eeggroup/CL_result/ > ~/CL_logs/cl_train_Sleep_r$r.$i.log
    python -u b_test.py --database_save_dir /data/eeggroup/CL_database/ --data_name Sleep --noise_ratio $r --exp_id $i --gpu_id 0 --load_path /data/eeggroup/CL_result/ > ~/CL_logs/cl_test_Sleep_r$r.$i.log
    wait
  done
  wait
	
	python -u b_test.py --data_name Sleep --noise_ratio $r --load_path /data/eeggroup/CL_result/ --summary True
  wait
done
wait


for r in 0 0.2 0.4
do
  for i in {1..6}
  do
    python -u a_train.py --database_save_dir /data/eeggroup/CL_database/ --data_name HHAR --noise_ratio $r --exp_id $i --gpu_id 0 --path_checkpoint /data/eeggroup/CL_result/ > ~/CL_logs/cl_train_HHAR_r$r.$i.log
    python -u b_test.py --database_save_dir /data/eeggroup/CL_database/ --data_name HHAR --noise_ratio $r --exp_id $i --gpu_id 0 --load_path /data/eeggroup/CL_result/ > ~/CL_logs/cl_test_HHAR_r$r.$i.log
    wait
  done
  wait

  python -u b_test.py --data_name HHAR --noise_ratio $r --load_path /data/eeggroup/CL_result/ --summary True
  wait
done
wait
