cd ../..


# 3 Students + OL using One
python driver.py --index 2 --tm resnet32x4 --sm vgg13 --use_hinton 1 --rk_type crd --fk_type semckd --ol One --gpu_id 3 

python driver.py --index 2 --tm vgg13 --sm ShuffleV2 --use_hinton 1 --rk_type crd --fk_type semckd --ol One --gpu_id 3 

python driver.py --index 2 --tm wrn_40_2 --sm mobilenetv2 --use_hinton 1 --rk_type crd --fk_type semckd --ol One --gpu_id 3 