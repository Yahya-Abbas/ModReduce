# cd ../..

# 3 Students + Asynchronous OL using FC (cutoff epoch 220, ol weight 0.7)
python driver.py --index 8 --tm wrn_40_2 --sm wrn_16_2 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm wrn_40_2 --sm wrn_40_1 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm resnet56 --sm resnet20 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm resnet110 --sm resnet20 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm resnet110 --sm resnet32 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm vgg13 --sm mobilenetv2 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm resnet32x4 --sm ShuffleV1 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm wrn_40_2 --sm ShuffleV1 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm resnet32x4 --sm resnet8x4 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm vgg13 --sm vgg8 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm resnet32x4 --sm ShuffleV2 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm resnet32x4 --sm vgg8 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm resnet32x4 --sm vgg13 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm vgg13 --sm ShuffleV2 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0

python driver.py --index 8 --tm wrn_40_2 --sm mobilenetv2 --use_hinton 1 --rk_type crd --fk_type semckd --ol FC --cutoff_epoch 220 --gpu_id 0