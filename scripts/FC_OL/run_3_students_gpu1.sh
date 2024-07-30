cd ../..

# 3 Students + OL using FC
python driver.py --index 3 --tm resnet110 --sm resnet32 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --gpu_id 1 

python driver.py --index 3 --tm vgg13 --sm mobilenetv2 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --gpu_id 1 

python driver.py --index 3 --tm resnet32x4 --sm ShuffleV1 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --gpu_id 1 

python driver.py --index 3 --tm wrn_40_2 --sm ShuffleV1 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --gpu_id 1 