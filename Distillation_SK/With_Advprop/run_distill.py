import os
import sys
cmd1 = "python train_teacher.py --model='resnet50'"
#cmd1 = "python train_student.py --path_t='save/models/resnet56_cifar100_lr_0.05_decay_0.0005_trial_0/resnet56_best.pth'\
#        --gamma=1 --alpha=0 --beta=1000 --model_s='resnet20' --distill='kd'"
os.system(cmd1)
print("completed run_distill")
