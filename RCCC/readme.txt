This repository provides the implementation for "Provably Consistent Partial-Label Learning" (NeurIPS 2020).

Requirements:
Python 3.6
numpy 1.14
Pytorch 1.1
torchvision 0.2

Demo:

	python main.py -lo rc -mo mlp -ds mnist -lr 1e-3 -wd 1e-5
	python main.py -lo rc -mo densenet -ds cifar10 -lr 1e-3 -wd 1e-3

-ds: specifies the dataset
-mo: specifies the model
-lo: specifies the loss function (method: rc or cc)
-lr: learning rate
-wd: weight decay

Thank you！