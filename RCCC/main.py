import argparse
from utils.models import mlp_model, linear_model, LeNet
import torch
from utils.utils_data import prepare_cv_datasets, prepare_train_loaders_for_uniform_cv_candidate_labels
from utils.utils_algo import accuracy_check, confidence_update
from utils.utils_loss import rc_loss, cc_loss
from cifar_models import densenet, resnet
import numpy as np

torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser()

parser.add_argument('-lr', help='optimizer\'s learning rate', default=1e-3, type=float)
parser.add_argument('-bs', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-ds', help='specify a dataset', default='mnist', type=str, required=False) # mnist, kmnist, fashion, cifar10
parser.add_argument('-mo', help='model name', default='mlp', choices=['linear', 'mlp', 'resnet', 'densenet', 'lenet'], type=str, required=False)
parser.add_argument('-ep', help='number of epochs', type=int, default=250)
parser.add_argument('-wd', help='weight decay', default=1e-5, type=float)
parser.add_argument('-lo', help='specify a loss function', default='rc', type=str, choices=['rc','cc'], required=False)
parser.add_argument('-seed', help = 'Random seed', default=0, type=int, required=False)
parser.add_argument('-gpu', help = 'used gpu id', default='0', type=str, required=False)

args = parser.parse_args()
np.random.seed(args.seed)

device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K = prepare_cv_datasets(dataname=args.ds, batch_size=args.bs)
partial_matrix_train_loader, train_data, train_givenY, dim = prepare_train_loaders_for_uniform_cv_candidate_labels(dataname=args.ds, 
                                                                                                                   full_train_loader=full_train_loader, 
                                                                                                                   batch_size=args.bs)

if args.lo == 'rc':
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])   #repeat train_givenY.shape[1] times in dim 1
    confidence = train_givenY.float()/tempY
    confidence = confidence.to(device)
    loss_fn = rc_loss
elif args.lo == 'cc':
    loss_fn = cc_loss
    
if args.mo == 'mlp':
    model = mlp_model(input_dim=dim, hidden_dim=500, output_dim=K)  #dim=28*28=784
elif args.mo == 'linear':
    model = linear_model(input_dim=dim, output_dim=K)
elif args.mo == 'lenet':
    model = LeNet(output_dim=K) #  linear,mlp,lenet are for MNIST-type datasets.
elif args.mo == 'densenet':
    model = densenet(num_classes=K)
elif args.mo == 'resnet':
    model = resnet(depth=32, num_classes=K) # densenet,resnet are for CIFAR-10.

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)

# test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)

# print('Epoch: 0. Te Acc: {}'.format(test_accuracy))

test_acc_list = []
train_acc_list = []

for epoch in range(args.ep):
    model.train()
    for i, (images, labels, true_labels, index) in enumerate(partial_matrix_train_loader):
        X, Y, index = images.to(device), labels.to(device), index.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        if args.lo == 'rc':
            average_loss = loss_fn(outputs, confidence, index)
        elif args.lo == 'cc':
            average_loss = loss_fn(outputs, Y.float())
        average_loss.backward()
        optimizer.step()
        if args.lo == 'rc':
            confidence = confidence_update(model, confidence, X, Y, index)
    model.eval()
    test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
    
    print('Epoch: {}. Te Acc: {}.'.format(epoch+1, test_accuracy))
 
    if epoch >= (args.ep-10):
        test_acc_list.extend([test_accuracy])
            
avg_test_acc = np.mean(test_acc_list)

print("Learning Rate:", args.lr, "Weight Decay:", args.wd)
print("Average Test Accuracy over Last 10 Epochs:", avg_test_acc)
