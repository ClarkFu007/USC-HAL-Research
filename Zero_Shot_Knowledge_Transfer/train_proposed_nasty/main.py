import torch
from solver import *
from utils.helpers import *


def main(args):
    """
        Run the experiment as many times as there
    are seeds given, and write the mean and std
    to as an empty file's name for cleaner logging
    """

    torch.backends.cudnn.enabled = True  # Original: True
    torch.backends.cudnn.benchmark = True

    if len(args.seeds) > 1:
        test_accs = []
        base_name = args.experiment_name
        for seed in args.seeds:
            print('\n\n----------- SEED {} -----------\n\n'.format(seed))
            from utils.helpers import set_torch_seeds
            set_torch_seeds(seed)
            args.experiment_name = os.path.join(base_name, base_name + '_seed' + str(seed))
            # The str() function converts values to a string form to be combined with other strings.
            """Main solver class to train and test the generator and student adversarially."""
            from solver import ZeroShotKTSolver
            solver = ZeroShotKTSolver(args)
            test_acc = solver.run()  ### From here! ###
            test_accs.append(test_acc)
        mu = np.mean(test_accs)
        sigma = np.std(test_accs)
        print('\n\nThe final mean test accuracy: {:02.2f} +/- {:02.2f}'.format(mu, sigma))
        file_name = "mean_final_test_acc_{:02.2f}_pm_{:02.2f}".format(mu, sigma)
        with open(os.path.join(args.log_directory_path, base_name, file_name), 'w+') as f:
            f.write("NA")
    else:
        from utils.helpers import set_torch_seeds
        set_torch_seeds(args.seeds[0])
        """Main solver class to train and test the generator and student adversarially."""
        from solver import ZeroShotKTSolver
        solver = ZeroShotKTSolver(args)
        test_acc = solver.run()
        print('\n\nThe final mean test accuracy: {:02.2f}'.format(test_acc))
        file_name = "final_test_acc_{:02.2f}".format(test_acc)
        with open(os.path.join(args.log_directory_path, args.experiment_name, file_name), 'w+') as f:
            f.write("NA")


if __name__ == "__main__":
    import argparse
    import time
    import numpy as np
    from utils.helpers import str2bool  # Used in argparse, to pass booleans

    print('Running...')
    time_start = time.time()

    parser = argparse.ArgumentParser(description='Welcome to the future')

    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['SVHN', 'CIFAR10'])
    parser.add_argument('--total_n_pseudo_batches', type=float, default=10000)
    parser.add_argument('--n_generator_iter', type=int, default=1, help='n_G gradient updates')
    parser.add_argument('--n_student_iter', type=int, default=10, help='n_S gradient updates')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='for each iteration of N, sample one batch of input z')
    parser.add_argument('--z_dim', type=int, default=100, help='the value of dimensions of input noise')
    parser.add_argument('--student_learning_rate', type=float, default=2e-3)
    parser.add_argument('--generator_learning_rate', type=float, default=1e-3)
    parser.add_argument('--teacher_architecture', type=str, default='ResNet34_nasty_loss2_withAtt')
    parser.add_argument('--student_architecture', type=str, default='ResNet18')
    parser.add_argument('--KL_temperature', type=float, default=1,
                        help='>1 to smooth probabilities in divergence loss, or <1 to sharpen them')
    parser.add_argument('--AT_beta', type=float, default=250, help='beta coefficient for attention term loss')

    parser.add_argument('--pretrained_models_path', nargs="?", type=str,
                        default='/home/yaof/trained_models/res34_nasty_teacher_cifar10')

    parser.add_argument('--datasets_path', type=str,
                        default="/home/yaof/data/cifar10")

    parser.add_argument('--log_directory_path', type=str,
                        default="/home/yaof/zero_shot/ZeroShot/ZeroShotKnowledgeTransfer/logs")
    parser.add_argument('--save_final_model', type=str2bool, default=True)
    parser.add_argument('--save_n_checkpoints', type=int, default=0)
    parser.add_argument('--save_model_path', type=str, default="/home/yaof/zero_shot/ZeroShot/FewShotKT/logs")

    parser.add_argument('--seeds', nargs='*', type=int, default=[0])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--use_gpu', type=str2bool, default=True,
                        help='set to False to debug on cpu, using LeNets')
    args = parser.parse_args()

    args.total_n_pseudo_batches = int(args.total_n_pseudo_batches)
    if args.AT_beta > 0:  # The beta coefficient for AT loss
        """
            The assert statement exists in almost every programming language. 
        It helps detect problems early in your program, where the cause is clear, 
        rather than later when some other operation fails.
            When you do "assert condition", you're telling the program to test that condition, 
        and immediately trigger an error if the condition is false.
        """
        assert args.student_architecture[:3] in args.teacher_architecture

    args.log_freq = max(1, int(args.total_n_pseudo_batches / 100))  # The frequency of logs

    # args.dataset_path = os.path.join(args.datasets_path, args.dataset)
    args.dataset_path = args.datasets_path

    args.use_gpu = args.use_gpu and torch.cuda.is_available()

    # args.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    args.device = 'cuda:0' if args.use_gpu else 'cpu'

    args.experiment_name = 'ZeroShotKnowledgeTransfer_{}_{}_{}_gi{}_si{}_zd{}_plr{}_slr{}_bs{}_T{}_beta{}' \
        .format(args.dataset, args.teacher_architecture, args.student_architecture,
                args.n_generator_iter, args.n_student_iter, args.z_dim, args.generator_learning_rate,
                args.student_learning_rate, args.batch_size, args.KL_temperature, args.AT_beta)

    print('\nTotal data batches: {}'.format(args.total_n_pseudo_batches))
    print('Logging results every {} batch'.format(args.log_freq))
    print('\nRunning on device: {}'.format(args.device))

    main(args)

    time_end = time.time()
    print("The total running time is %f seconds" % (time_end - time_start))
