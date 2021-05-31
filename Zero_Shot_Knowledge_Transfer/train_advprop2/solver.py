import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import shutil
from time import time
from utils.logger import *
from utils.datasets import *
from models.selector import *
from utils.loaders import *


class MixBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input


class ZeroShotKTSolver(object):
    """
        Main solver class to train and test the generator and student adversarially.
    """
    def __init__(self, args):  # The constructor in Python
        self.args = args

        ## Student and Teacher Nets
        from attacker import NoOpAttacker, PGDAttacker
        attack_iter, attack_epsilon, attack_step_size = 4, 5.0, 1.0
        if attack_iter == 0:
            attacker = NoOpAttacker()
        else:
            attacker = PGDAttacker(attack_iter, attack_epsilon, attack_step_size,
                                   prob_start_from_clean=0.2)
        norm_layer = MixBatchNorm2d

        arch = "wide_resnet50_2"
        import net_rectified
        self.teacher = net_rectified.__dict__[arch](num_classes=10, norm_layer=norm_layer)
        self.teacher.set_attacker(attacker)
        self.teacher.set_mixbn(False)
        self.teacher = torch.nn.DataParallel(self.teacher)
        self.teacher = self.teacher.to(args.device)

        model_path = args.pretrained_models_path
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        # self.teacher.load_state_dict(checkpoint)
        mixbn = False  # Whether to make the teacher have mixbn or not
        if mixbn:
            to_merge = {}
            for key in checkpoint['state_dict']:
                if 'bn' in key:
                    tmp = key.split("bn")
                    aux_key = tmp[0] + 'bn' + tmp[1][0] + '.aux_bn' + tmp[1][1:]
                    to_merge[aux_key] = checkpoint['state_dict'][key]
                elif 'downsample.1' in key:
                    tmp = key.split("downsample.1")
                    aux_key = tmp[0] + 'downsample.1.aux_bn' + tmp[1]
                    to_merge[aux_key] = checkpoint['state_dict'][key]
            checkpoint['state_dict'].update(to_merge)
        self.teacher.load_state_dict(checkpoint['state_dict'],
                                     strict=False)  # "Strict" can handle DataParallel problem
        self.teacher.eval()

        from models.wresnet import WideResNet
        self.student = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
        self.student = self.student.to(args.device)
        self.student.train()

        ## Loaders
        self.n_repeat_batch = args.n_generator_iter + args.n_student_iter  # n_G + n_S
        from utils.loaders import LearnableLoader
        """Infinite loader, which contains a learnable generator."""
        self.generator = LearnableLoader(args=args,
                                         n_repeat_batch=self.n_repeat_batch).to(device=args.device)
        from utils.datasets import get_test_loader
        self.test_loader = get_test_loader(args)  # Get real test images

        ## Optimizers & Schedulers
        import torch.optim as optim
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=args.generator_learning_rate)
        self.scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_generator,
                                                                        args.total_n_pseudo_batches,
                                                                        last_epoch=-1)
        self.optimizer_student = optim.Adam(self.student.parameters(), lr=args.student_learning_rate)
        self.scheduler_student = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student,
                                                                      args.total_n_pseudo_batches,
                                                                      last_epoch=-1)

        ### Set up & Resume
        self.n_pseudo_batches = 0
        self.experiment_path = os.path.join(args.log_directory_path, args.experiment_name)
        self.save_model_path = os.path.join(args.save_model_path, args.experiment_name)
        from utils.logger import Logger
        self.logger = Logger(log_dir=self.experiment_path)

        if os.path.exists(self.experiment_path):
            if self.args.use_gpu:
                checkpoint_path = os.path.join(self.experiment_path, 'last.pth.tar')
                if os.path.isfile(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)
                    print('\nResuming from checkpoint file at batch iter {} with top 1 acc {}\n'
                          .format(checkpoint['n_pseudo_batches'], checkpoint['test_acc']))
                    print('Running an extra {} iterations'
                          .format(args.total_n_pseudo_batches - checkpoint['n_pseudo_batches']))
                    self.n_pseudo_batches = checkpoint['n_pseudo_batches']
                    self.generator.load_state_dict(checkpoint['generator_state_dict'])
                    self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
                    self.scheduler_generator.load_state_dict(checkpoint['scheduler_generator'])
                    self.student.load_state_dict(checkpoint['student_state_dict'])
                    self.optimizer_student.load_state_dict(checkpoint['optimizer_student'])
                    self.scheduler_student.load_state_dict(checkpoint['scheduler_student'])
            else:
                shutil.rmtree(self.experiment_path)  # clear debug logs on cpu
                os.makedirs(self.experiment_path)
        else:
            os.makedirs(self.experiment_path)

        ## Save and Print Args
        print('\n---------')
        with open(os.path.join(self.experiment_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')

    def run(self):
        from utils.helpers import AggregateScalar
        # torch.backends.cudnn.enabled = False

        """Computes and stores the average and std of stream."""
        running_data_time, running_batch_time = AggregateScalar(), AggregateScalar()
        running_student_maxes_avg, running_teacher_maxes_avg = AggregateScalar(), AggregateScalar()
        running_student_total_loss, running_generator_total_loss = AggregateScalar(), AggregateScalar()
        student_maxes_distribution, student_argmaxes_distribution = [], []
        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []

        end = time()
        idx_pseudo = 0
        # For 1, 2, ..., N
        while self.n_pseudo_batches < self.args.total_n_pseudo_batches:
            x_pseudo = self.generator.__next__()
            running_data_time.update(time() - end)

            ## Take n_generator_iter steps on generator
            if idx_pseudo % self.n_repeat_batch < self.args.n_generator_iter:
                # May be two more output activations
                student_logits, *student_activations = self.student(x_pseudo)
                teacher_logits, *teacher_activations = self.teacher(x_pseudo)
                # print("The iteration is", idx_pseudo)
                # print("student_logits.shape", student_logits.shape)
                # print("*student_activations", type(*student_activations))
                # print("teacher_logits.shape", teacher_logits.shape)
                # print("*teacher_activations", type(*teacher_activations))

                # Maximize D_kl for the generator loss without an attentation term
                generator_total_loss = self.KT_loss_generator(student_logits, teacher_logits)
                # torch.backends.cudnn.enabled = False
                self.optimizer_generator.zero_grad()
                generator_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
                self.optimizer_generator.step()

            ## Take n_student_iter steps on student
            elif idx_pseudo % self.n_repeat_batch < (self.args.n_generator_iter + self.args.n_student_iter):
                if idx_pseudo % self.n_repeat_batch == self.args.n_generator_iter:
                    with torch.no_grad():
                        # only need to calculate teacher logits once because teacher & x_pseudo fixed
                        teacher_logits, *teacher_activations = self.teacher(x_pseudo)

                student_logits, *student_activations = self.student(x_pseudo)

                # Minimize D_kl for the student loss with an attentation term
                student_total_loss = self.KT_loss_student(student_logits, student_activations,
                                                          teacher_logits, teacher_activations)
                self.optimizer_student.zero_grad()
                student_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 5)
                self.optimizer_student.step()

            ## Last call to this batch, log metrics

            if (idx_pseudo + 1) % self.n_repeat_batch == 0:
                with torch.no_grad():
                    # print("teacher_logits.shape", teacher_logits.shape)
                    teacher_maxes, teacher_argmaxes = torch.max(torch.softmax(teacher_logits, dim=1), dim=1)
                    student_maxes, student_argmaxes = torch.max(torch.softmax(student_logits, dim=1), dim=1)
                    running_generator_total_loss.update(float(generator_total_loss))
                    running_student_total_loss.update(float(student_total_loss))
                    running_teacher_maxes_avg.update(float(torch.mean(teacher_maxes)))
                    running_student_maxes_avg.update(float(torch.mean(student_maxes)))
                    teacher_maxes_distribution.append(teacher_maxes)
                    teacher_argmaxes_distribution.append(teacher_argmaxes)
                    student_maxes_distribution.append(student_maxes)
                    student_argmaxes_distribution.append(student_argmaxes)

                if (self.n_pseudo_batches+1) % self.args.log_freq == 0:
                    test_acc = self.test()

                    with torch.no_grad():
                        print('\nBatch {}/{} -- Generator Loss: {:02.2f} -- Student Loss: {:02.2f}'
                              .format(self.n_pseudo_batches, self.args.total_n_pseudo_batches,
                                      running_generator_total_loss.avg(), running_student_total_loss.avg()))
                        print('Test Acc: {:02.2f}%'.format(test_acc*100))
                        """
                        self.logger.scalar_summary('TRAIN_PSEUDO/generator_total_loss',
                                                   running_generator_total_loss.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/student_total_loss',
                                                   running_student_total_loss.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/teacher_maxes_avg',
                                                   running_teacher_maxes_avg.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/student_maxes_avg',
                                                   running_student_maxes_avg.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/student_lr',
                                                   self.scheduler_student.get_lr()[0], self.n_pseudo_batches)
                        self.logger.scalar_summary('TRAIN_PSEUDO/generator_lr',
                                                   self.scheduler_generator.get_lr()[0], self.n_pseudo_batches)
                        self.logger.scalar_summary('TIME/data_time_sec',
                                                   running_data_time.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('TIME/batch_time_sec',
                                                   running_batch_time.avg(), self.n_pseudo_batches)
                        self.logger.scalar_summary('EVALUATE/test_acc',
                                                   test_acc*100, self.n_pseudo_batches)
                        self.logger.image_summary('RANDOM', self.generator.samples(n=9, grid=True),
                                                  self.n_pseudo_batches)
                        # print("teacher_maxes_distribution.shape", teacher_maxes_distribution)
                        # print("teacher_argmaxes_distribution.shape", len(teacher_argmaxes_distribution))
                        # print("student_maxes_distribution.shape", len(student_maxes_distribution))
                        # print("student_argmaxes_distribution.shape", len(student_argmaxes_distribution))
                        self.logger.histo_summary('TEACHER_MAXES_DISTRIBUTION',
                                                  torch.cat(teacher_maxes_distribution),
                                                  self.n_pseudo_batches)
                        self.logger.histo_summary('TEACHER_ARGMAXES_DISTRIBUTION',
                                                  torch.cat(teacher_argmaxes_distribution),
                                                  self.n_pseudo_batches)
                        self.logger.histo_summary('STUDENT_MAXES_DISTRIBUTION',
                                                  torch.cat(student_maxes_distribution),
                                                  self.n_pseudo_batches)
                        self.logger.histo_summary('STUDENT_ARGMAXES_DISTRIBUTION',
                                                  torch.cat(student_argmaxes_distribution),
                                                  self.n_pseudo_batches)
                        self.logger.write_to_csv('train_test.csv')
                        self.logger.writer.flush()
                        
                        running_data_time.reset(), running_batch_time.reset()
                        running_teacher_maxes_avg.reset(), running_student_maxes_avg.reset()
                        running_generator_total_loss.reset(), running_student_total_loss.reset(),
                        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
                        student_maxes_distribution, student_argmaxes_distribution = [], []
                        """

                if self.args.save_n_checkpoints > 1:
                    if (self.n_pseudo_batches+1) % \
                            int(self.args.total_n_pseudo_batches / self.args.save_n_checkpoints) == 0:
                        test_acc = self.test()
                        self.save_model(test_acc=test_acc)

                self.n_pseudo_batches += 1
                import torchvision.utils as vutils
                """
                if self.n_pseudo_batches <= 5000 and self.n_pseudo_batches % 50 == 0:
                    vutils.save_image(x_pseudo.data.clone(),
                                      '/home/yaof/ZeroShot/temp_images/output_{}.png'
                                      .format(self.n_pseudo_batches), normalize=True,
                                      scale_each=True, nrow=8)
                
                                if self.n_pseudo_batches % 500 == 0:
                    vutils.save_image(x_pseudo.data.clone(),
                                      '/home/yaof/ZeroShot/temp_images/output_{}.png'
                                      .format(self.n_pseudo_batches), normalize=True,
                                      scale_each=True, nrow=8)
                """

                # print("The pseudo_batch %d is finished!" % (self.n_pseudo_batches))
                self.scheduler_student.step()
                self.scheduler_generator.step()

            # print("The iteration %d is finished!" % (idx_pseudo))
            idx_pseudo += 1

            running_batch_time.update(time() - end)
            end = time()

        test_acc = self.test()
        if self.args.save_final_model:  # make sure last epoch saved
            self.save_model(test_acc=test_acc)

        return test_acc*100

    def test(self):

        self.student.eval()
        running_test_acc = AggregateScalar()

        from utils.helpers import accuracy
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.args.device), y.to(self.args.device)
                student_logits, *student_activations = self.student(x)
                # print("student_logits.data.shape", student_logits.data.shape) # [128,10]
                # print("y.shape", y.shape) # [128]
                # The one-element tuple
                acc = accuracy(student_logits.data, y, topk=(1,))[0]
                running_test_acc.update(float(acc), x.shape[0])

        self.student.train()
        return running_test_acc.avg()

    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()

    def divergence(self, student_logits, teacher_logits):

        divergence = F.kl_div(F.log_softmax(student_logits / self.args.KL_temperature, dim=1),
                              F.softmax(teacher_logits / self.args.KL_temperature, dim=1))  # forward KL
        return divergence

    def KT_loss_generator(self, student_logits, teacher_logits):

        divergence_loss = self.divergence(student_logits, teacher_logits)
        total_loss = - divergence_loss

        return total_loss

    def KT_loss_student(self, student_logits, student_activations, teacher_logits, teacher_activations):

        divergence_loss = self.divergence(student_logits, teacher_logits)
        if self.args.AT_beta > 0:
            at_loss = 0
            for i in range(len(student_activations)):
                at_loss = at_loss + self.args.AT_beta * \
                          self.attention_diff(student_activations[i], teacher_activations[i])
        else:
            at_loss = 0

        total_loss = divergence_loss + at_loss

        return total_loss

    def save_model(self, test_acc):

        delete_files_from_name(self.save_model_path, "test_acc_", type='contains')
        file_name = "n_batches_{}_test_acc_{:02.2f}".format(self.n_pseudo_batches, test_acc * 100)
        with open(os.path.join(self.save_model_path, file_name), 'w+') as f:
            f.write("NA")

        torch.save({'args': self.args,
                    'n_pseudo_batches': self.n_pseudo_batches,
                    'generator_state_dict': self.generator.state_dict(),
                    'student_state_dict': self.student.state_dict(),
                    'optimizer_generator': self.optimizer_generator.state_dict(),
                    'optimizer_student': self.optimizer_student.state_dict(),
                    'scheduler_generator': self.scheduler_generator.state_dict(),
                    'scheduler_student': self.scheduler_student.state_dict(),
                    'test_acc': test_acc},
                   os.path.join(self.save_model_path, "last.pth.tar"))
        print("\nSaved model with test acc {:02.2f}%\n".format(test_acc * 100))