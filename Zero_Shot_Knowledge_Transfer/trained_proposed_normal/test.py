import torch

checkpoint_path1 = "ZeroShotKT_CIFAR10_ResNet34_nasty_loss2_withAtt_ResNet18_best.pth.tar"
checkpoint1 = torch.load(checkpoint_path1)
print('\nThe accuracy is {}\n'.format(checkpoint1['test_acc']))

checkpoint_path2 = "ZeroShotKT_CIFAR10_ResNet34_nasty_loss2_withAtt_ResNet18_last.pth.tar"
checkpoint2 = torch.load(checkpoint_path2)
print('\nThe accuracy is {}\n'.format(checkpoint2['test_acc']))
