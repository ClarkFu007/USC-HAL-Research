# pytorch-ZeroShot-experiments
A PyTorch implementation of ZeroShot experiments.

## Environments
This project is developed and tested in the following environments.
* Ubuntu 18.04
* CUDA 10.2
* GTX 1070 TI
* Python 3.8.1

# Implementation
* train_advprop1:
    We can use both the set of BN and Aux BNs of the teacher model to generate synthetic images of two distributions.
If we do so, then instead of n images now, we can have 2n images of same label. If this is the case, we can now 
have more synthetic images for the student to be trained. So, we need to check whether the student can perform 
better with data from two distribution. For this case, I think the student training should also use two BN and 
Aux BN as based on which set of synthetic images we are using (whether from distri. 1 or distri 2).
  
* train_advprop2:
    We can only use one BN of the teacher model and consider that as the only BN present in the teacher 
(by ignoring the AuxBN) and generate synthetic images with that single BN based teacher and use those 
images for the training of the student. Now, we need to check does it improve the student’s learning 
from the teacher or degrade the performance. 

* train_normal:
    trained with single BN based model, on only clean images.

Another interesting experiment can be, if I have access of the BN and AuxBN stat of the teacher, 
then can I use a mixup BN statistics for the teacher and use that as new BN?  I mean, suppose we have u1, 
sigma1 and u2, signam2 as BN and auxBN running mean and variance values. In that case, can we use a mixed 
umix = alpha*u1 + (1-alpha)u2, and sigma_mix = alpha*sigma1+(1-alpha)*sigma2. And use this as new BN of the 
teacher model, and then generate the synthetic images. For this experiment we need to change the internal 
running mean and variance of the BN of teacher by the new formula.


