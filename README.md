# PAGDNet
Pixel attention-guided CNN for image denoising（PAGDNet）by Nanrun Zhou, Jibin Deng, Wenjun Yu and Meng Pang is submitted in Knowledge-Based Systems, 2023.

# Prerequisites:

python == 3.6.2

tensorflow == 2.0.0

keras == 2.0.9

opencv-python == 4.5.5.62

scikit-image == 0.17.2

# Denoising Training
For train the PAGDNet, please run:

python mainimprovement.py

# Denoising Testing
For test the PAGDNet, please run:

python mainimprovement.py --pretrain sigma/model_50.h5 --only_test True

# Denoising Datasets
The gray train dataset "Train400" you can download here (Selected in the paper):

https://download.csdn.net/download/qq_41104871/87646484

The color train dataset "BSD400" you can download here (Selected in the paper):

https://download.csdn.net/download/qq_41104871/87647333

The real-world train dataset "PolyU" you can download here (Selected in the paper):

https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset

The real-world test dataset "CC" you can download here (Selected in the paper):

https://github.com/csjunxu/MCWNNM_ICCV2017

The real-world test dataset "Nam" you can download here (Selected in the paper):

https://github.com/GuoShi28/CBDNet/tree/master/testsets/Nam_patches

The real-world test dataset "DND" you can download here (No ground truth):

https://github.com/GuoShi28/CBDNet/tree/master/testsets/DND_patches