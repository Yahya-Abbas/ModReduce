cd ../..

# 3 Students + OL using pcl
python driver.py --index 1 --tm resnet32x4 --sm resnet8x4 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 2 

python driver.py --index 1 --tm vgg13 --sm vgg8 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 2 

python driver.py --index 1 --tm resnet32x4 --sm ShuffleV2 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 2 

python driver.py --index 1 --tm resnet32x4 --sm vgg8 --use_hinton 1 --rk_type crd --fk_type semckd --ol pcl --gpu_id 2 