"""
Name: Sagi Della Torre
Date: 17/02/2023
Training procedure for semi_supervised_vae.
"""
import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
from m2_model import M2,Classifier

# prepare dataloader for pytorch
class TorchInputData(torch.utils.data.Dataset):
    """
    A simple inheretance of torch.DataSet to enable using our customized DogBreed dataset in torch
    """
    def __init__(self, X, Y, transform=None):
        """
        X: a list of numpy images 
        Y: a list of labels coded using 0-9 
        """        
        self.X = X
        self.Y = Y 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        return x, y

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
    ])

    unlabeled_proportion = args.unlabeled_proportion

    data_len = 10000
    labled_data_len = int(data_len * unlabeled_proportion / 100)
    test_len = int(labled_data_len * 0.3)

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data',
            train=True, download=True, transform=transform)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data',
            train=True, download=True, transform=transform)
    else:
        raise ValueError('Dataset not implemented')

    # take 10000 images, and devide the data into labeled, and unlabled
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=data_len, shuffle=True, num_workers=2)

    # devide the data into labeled, and unlabled
    data, labels= next(iter(trainloader))

    # choose 1500 from 10000 indexes
    np.random.seed(5)
    labeled_ind = np.random.choice(data_len,labled_data_len, replace = False)
    unlabeled_ind = np.setdiff1d(list(range(data_len)), labeled_ind)

    # label of unlabel data coded as 10
    labels = labels.numpy()
    np.put(labels,list(unlabeled_ind),10)

    # devide all the data to train (from labeled and unlabeled) and test (from the labeled data only)
    test_ind = labeled_ind[np.random.choice(labled_data_len,test_len, replace = False)]
    train_ind = np.setdiff1d(list(range(data_len)), test_ind)

    # create the trainloader and testloader
    images_train = [data[i] for i in train_ind]
    trainset = TorchInputData(images_train, labels[train_ind])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    images_dev = [data[i] for i in test_ind]
    testset = TorchInputData(images_dev, labels[test_ind])
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    # the models
    classifier = Classifier(image_reso = 28, filter_size = 5, dropout_rate = 0.2)
    m2 = M2(latent_features = args.latent_dim, classifier = classifier, dataset = args.dataset, unlabeled_proportion=unlabeled_proportion, path = "model/m2_{}_unlabeled_prop_{}_epoch_{}.pth".format(args.dataset, args.unlabeled_proportion, args.epochs))

    #set alpha, hyperparameter for weighing the classifier loss
    alpha = 0.1*len(trainloader.dataset)

    # fit M2 model
    # labeled_data_len is the number of labeled data in train+test set: 450+1050
    m2.fit(trainloader, testloader, args.batch_size, alpha, labeled_data_len = labled_data_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=50)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--unlabeled_proportion',
                        help='proportion between the labled date to the unlabled (Percents).',
                        type=int,
                        default=15)
    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=128)

    args = parser.parse_args()
    main(args)
