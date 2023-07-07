import os

import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import argparse

from distil.utils.dataset import AffectDataset

sys.path.append('./')
import torch
from torch.utils.data import Subset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
from distil.utils.models.resnet import ResNet18
from distil.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, LeastConfidenceSampling, \
    MarginSampling, CoreSet, AdversarialBIM, AdversarialDeepFool, KMeansSampling, \
    BALDDropout, FASS
from distil.utils.models.simple_net import TwoLayerNet
from distil.utils.train_helper import data_train
from distil.utils.config_helper import read_config_file
from distil.utils.utils import LabeledToUnlabeledDataset
import time
import pickle


class TrainClassifier:

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = read_config_file(config_file)
        os.makedirs(self.config['save_dir'], exist_ok=True)

    def getModel(self, model_config):

        if model_config['architecture'] == 'resnet18':

            if ('target_classes' in model_config) and ('channel' in model_config):
                net = ResNet18(num_classes=model_config['target_classes'], channels=model_config['channel'])
            elif 'target_classes' in model_config:
                net = ResNet18(num_classes=model_config['target_classes'])
            else:
                net = ResNet18()
        if 'ispretrained' in model_config:
            if model_config['ispretrained']:
                state_dict = torch.load(model_config['pretrained_model_path'])
                if 'model' in state_dict.keys():
                    state_dict = state_dict['model']
                else:
                    from torchvision.models import resnet18
                    net = resnet18(num_classes=model_config['target_classes'])
                    state_dict = state_dict['state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("encoder.", "")
                    k = k.replace("backbone.", "")
                    if not k.startswith('linear'):
                        new_state_dict[k] = v
                state_dict = new_state_dict
                try:
                    msg = net.load_state_dict(state_dict, strict=False)
                except:
                    net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    msg = net.load_state_dict(state_dict, strict=False)
                print(msg)

        elif model_config['architecture'] == 'two_layer_net':
            net = TwoLayerNet(model_config['input_dim'], model_config['target_classes'], model_config['hidden_units_1'])

        return net

    def libsvm_file_load(self, path, dim, save_data=False):

        data = []
        target = []
        with open(path) as fp:
            line = fp.readline()
            while line:
                temp = [i for i in line.strip().split(" ")]
                target.append(int(float(temp[0])))  # Class Number. # Not assumed to be in (0, K-1)
                temp_data = [0] * dim

                for i in temp[1:]:
                    ind, val = i.split(':')
                    temp_data[int(ind) - 1] = float(val)
                data.append(temp_data)
                line = fp.readline()
        X_data = np.array(data, dtype=np.float32)
        Y_label = np.array(target)
        if save_data:
            # Save the numpy files to the folder where they come from
            data_np_path = path + '.data.npy'
            target_np_path = path + '.label.npy'
            np.save(data_np_path, X_data)
            np.save(target_np_path, Y_label)

        return (X_data, Y_label)

    def getData(self, data_config):

        img_size = 48 if data_config['name'] == 'FER13' else 96 if data_config['name'] == 'RAF' else 112

        train_transform = transforms.Compose(
            [transforms.RandomCrop(img_size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

        train_dataset = AffectDataset(self.config['dataset']['dir'], train=True, download=True,
                                      transform=train_transform)
        test_dataset = AffectDataset(self.config['dataset']['dir'], train=False, download=True,
                                     transform=test_transform)

        return train_dataset, test_dataset

    def write_logs(self, logs, save_location, rd):

        file_path = save_location
        with open(file_path, 'a') as f:
            f.write('---------------------\n')
            f.write('Round ' + str(rd) + '\n')
            f.write('---------------------\n')
            for key, val in logs.items():
                if key == 'Training':
                    f.write(str(key) + '\n')
                    for epoch in val:
                        f.write(str(epoch) + '\n')
                else:
                    f.write(str(key) + ' - ' + str(val) + '\n')

    def train_classifier(self):

        net = self.getModel(self.config['model'])
        full_train_dataset, test_dataset = self.getData(self.config['dataset'])
        selected_strat = self.config['active_learning']['strategy']
        budget = self.config['active_learning']['budget']
        start = self.config['active_learning']['initial_points']
        n_rounds = self.config['active_learning']['rounds']
        nclasses = self.config['model']['target_classes']
        strategy_args = self.config['active_learning']['strategy_args']

        nSamps = len(full_train_dataset)
        np.random.seed(self.config['train_parameters']['seed'])
        start_idxs = np.random.choice(nSamps, size=start, replace=False)
        train_dataset = Subset(full_train_dataset, start_idxs)
        unlabeled_dataset = Subset(full_train_dataset, list(set(range(len(full_train_dataset))) - set(start_idxs)))

        if 'islogs' in self.config['train_parameters']:
            islogs = self.config['train_parameters']['islogs']
            save_location = self.config['train_parameters']['logs_location']
        else:
            islogs = False

        dt = data_train(train_dataset, net, self.config['train_parameters'])

        if selected_strat == 'badge':
            strategy = BADGE(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'glister':
            strategy = GLISTER(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                               strategy_args, validation_dataset=None, \
                               typeOf='Diversity', lam=10)
        elif selected_strat == 'entropy_sampling':
            strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                                       strategy_args)
        elif selected_strat == 'margin_sampling':
            strategy = MarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                                      strategy_args)
        elif selected_strat == 'least_confidence':
            strategy = LeastConfidenceSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net,
                                               nclasses, strategy_args)
        elif selected_strat == 'coreset':
            strategy = CoreSet(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                               strategy_args)
        elif selected_strat == 'fass':
            strategy = FASS(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)
        elif selected_strat == 'random_sampling':
            strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                                      strategy_args)
        elif selected_strat == 'bald_dropout':
            strategy = BALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                                   strategy_args)
        elif selected_strat == 'adversarial_bim':
            strategy = AdversarialBIM(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                                      strategy_args)
        elif selected_strat == 'kmeans_sampling':
            strategy = KMeansSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                                      strategy_args)
        elif selected_strat == 'adversarial_deepfool':
            strategy = AdversarialDeepFool(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses,
                                           strategy_args)
        else:
            raise IOError('Enter Valid Strategy!')

        if islogs:
            clf, train_logs = dt.train()
        else:
            clf = dt.train()
        strategy.update_model(clf)

        acc = np.zeros(n_rounds)
        acc[0] = dt.get_acc_on_set(test_dataset)

        if islogs:
            logs = {}
            logs['Training Points'] = len(train_dataset)
            logs['Test Accuracy'] = str(round(acc[0] * 100, 2))
            logs['Training'] = train_logs
            self.write_logs(logs, save_location, 0)

        print('***************************')
        print('Starting Training..')
        print('***************************')
        ##User Controlled Loop
        for rd in range(1, n_rounds):
            print('***************************')
            print('Round', rd)
            print('***************************')
            logs = {}
            t0 = time.time()
            idx = strategy.select(budget)
            t1 = time.time()

            # Adding new points to training set
            train_dataset = ConcatDataset([train_dataset, Subset(unlabeled_dataset, idx)])
            remain_idx = list(set(range(len(unlabeled_dataset))) - set(idx))
            unlabeled_dataset = Subset(unlabeled_dataset, remain_idx)

            print('Total training points in this round', len(train_dataset))

            strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
            dt.update_data(train_dataset)

            if islogs:
                clf, train_logs = dt.train()
            else:
                clf = dt.train()
            t2 = time.time()
            strategy.update_model(clf)
            acc[rd] = dt.get_acc_on_set(test_dataset)
            print('Testing Accuracy:', acc[rd])

            if islogs:
                logs['Training Points'] = len(train_dataset)
                logs['Test Accuracy'] = str(round(acc[rd] * 100, 2))
                logs['Selection Time'] = str(t1 - t0)
                logs['Trainining Time'] = str(t2 - t1)
                logs['Training'] = train_logs
                self.write_logs(logs, save_location, rd)

        print('Training Completed!')
        with open('./final_model.pkl', 'wb') as save_file:
            pickle.dump(clf.state_dict(), save_file)
        print('Model Saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help="Path to the config file")
    args = parser.parse_args()

    tc = TrainClassifier(args.config_path)
    tc.train_classifier()
