cd ../..

# 3 Students + OL using pcl
python driver.py --index 1 --tm wrn_40_2 --sm wrn_16_2 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 0 

python driver.py --index 1 --tm wrn_40_2 --sm wrn_40_1 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 0 

python driver.py --index 1 --tm resnet56 --sm resnet20 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 0 

python driver.py --index 1 --tm resnet110 --sm resnet20 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 0 