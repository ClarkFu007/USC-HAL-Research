import os
import sys
#cmd1 = "python train_teacher.py --model='resnet50'"
cmd1 = "python train_student.py --path_t='save/models/resnet50_cifar100_lr_0.05_decay_0.0005_trial_0/resnet50_best.pth'\
        --gamma=1.0 --alpha=0 --beta=1000 --model_s='sa_resnet26' --distill='gram_stem' --batch_size=32 --self_attn=True\
        --lr_decay_epochs='120,160,180' --epochs=200 --learning_rate=0.05 --weight_decay=1e-4 --stem True --dataset 'cifar100'"
os.system(cmd1)
print("completed run_distill_sa")
