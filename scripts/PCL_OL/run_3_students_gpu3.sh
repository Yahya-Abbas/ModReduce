cd ../..

# 3 Students + OL using pcl
python driver.py --index 1 --tm resnet32x4 --sm vgg13 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 3 

python driver.py --index 1 --tm vgg13 --sm ShuffleV2 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 3 

python driver.py --index 1 --tm wrn_40_2 --sm mobilenetv2 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 3 