import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
from .helper import generate_uniform_cv_candidate_labels
from PIL import Image
import torch.nn.functional as F

class CIFAR10(dsets.CIFAR10):
    def __init__(self, root, data_split, transform_mean, transform_std, img_size=32, pp=1, annFile="", label_mask=None, partial=1+1e-6):
        self.root = root
        self.data_split = data_split
        train_transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32,4),
            transforms.Normalize(transform_mean, transform_std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)])
        # self.classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]   #TODO if the "s" is important
        # CIFAR10 has attr self.classes

        if self.data_split == 'train':
            self.transform = train_transform
        elif self.data_split == 'test':
            self.transform = test_transform
        else:
            raise ValueError('data split = %s is not supported in Nus Wide' % self.data_split)
        
        super(CIFAR10, self).__init__(root=self.root, train=self.data_split == 'train', transform=self.transform, download=True) #debug the error in this line :"RuntimeError: super(): no arguments" TOArx

        if self.data_split == 'train':
            self.reinit_labels(partial_portion=pp)     
        # else:
        #     self.targets = F.one_hot(torch.tensor(self.targets), num_classes=len(self.classes)).float()



    def build_dataloader(self, batch_size, shuffle, num_workers=15):
        loader = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader
                                             

    def reinit_labels(self, partial_portion):
        partialY = generate_uniform_cv_candidate_labels(torch.Tensor(self.targets).long(), partial_portion)
        # temp = torch.zeros(partialY.shape)
        # temp[torch.arange(partialY.shape[0]), torch.Tensor(self.targets).long()] = 1
        # if torch.sum(partialY * temp) == partialY.shape[0]:
        #     print('partialY correctly loaded')
        # else:
        #     print('inconsistent permutation')

        self.targets_old = self.targets
        self.targets = partialY.float()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target, index

    def name(self):
        return 'CIFAR10'

