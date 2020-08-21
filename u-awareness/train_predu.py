"""
Modified by:
Author: Tianwei Xing (twxing@ucla.edu)
"""

import multiprocessing
import pickle
import sys
import argparse
import os
import time

import numpy as np
from numpy import savetxt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CLEVR, collate_data, transform, GQA
from src.model_predu import MACNetwork
# from model_gqa import MACNetwork

batch_size = 128
learning_rate = 1e-4
dim_dict = {'CLEVR': 512,
            'gqa': 2048}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decay', type=float, default=0.999, help='Running average: decay rate for updating params, default 0.999. (If set to zero, no RA)')
    parser.add_argument("--load_embd", type=bool, default=False, help="If loading pre-trained GLOVE embedding, default is False")
    parser.add_argument("--out_name", type=str, default='try', help="output directory, default \'try\'")
    parser.add_argument("--epoch", type=int, default=25, help="Number of training epochs default 25")
    
    arguments = parser.parse_args()
    return arguments


# params updating using running average 
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch, dataset_type, dataset_object):

    train_set = DataLoader(
        dataset_object, batch_size=batch_size, num_workers=1, collate_fn=collate_data
#         dataset_object, batch_size=batch_size, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0
    
    log_softmax = nn.LogSoftmax().cuda()
    aleatoric_loss = AleatoricCrossEntropyLoss().cuda()

    net.train(True)
    for iter_id, (image, question, q_len, answer) in enumerate(pbar):
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )
        y = one_hot_embedding(answer, 28)
        y = y.to(device)
        
        net.zero_grad()
        output, logits_variance = net(image, question, q_len)
        
        nll = -log_softmax(output)
        ud_ce_loss = (nll * y / 10).sum(dim=1).mean()
        
        gce_loss, variance_loss, undistorted_loss, variance_depressor = aleatoric_loss(logits_variance, output, y)
        aleatoric_uncertainty_loss = gce_loss + variance_loss + variance_depressor
        loss = ud_ce_loss + aleatoric_uncertainty_loss
        
        loss.backward(retain_graph=True)
        aleatoric_uncertainty_loss.backward(retain_graph=True)  
        # in multi-task learning scenario, have to choose "retain_graph=True" before performing multiple "backward"
        
        optimizer.step()

        correct = output.detach().argmax(1) == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct
        else:
            moving_loss = (moving_loss * iter_id + correct) / (iter_id + 1)

        pbar.set_description('Epoch: {}; CurLoss: {:.8f}; CurAcc: {:.5f}; Tot_Acc: {:.5f}'.format(epoch + 1, loss.item(), correct, moving_loss))

        accumulate(net_running, net, decay)
#     dataset_object.close()
    return



def valid(epoch, dataset_type, dataset_object):

    valid_set = DataLoader(
        dataset_object, batch_size=4*batch_size, num_workers=1,  collate_fn=collate_data
#         dataset_object, batch_size=4*batch_size, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
    )
    dataset = iter(valid_set)

    net_running.train(False)
    correct_counts = 0
    total_counts = 0
    running_loss = 0.0
    batches_done = 0
    
    log_softmax = nn.LogSoftmax().cuda()
    aleatoric_loss = AleatoricCrossEntropyLoss().cuda()
    
    with torch.no_grad():
        pbar = tqdm(dataset)
        for image, question, q_len, answer in pbar:
            image, question, answer = (
                image.to(device),
                question.to(device),
                answer.to(device),
            )

            y = one_hot_embedding(answer, 28)
            y = y.to(device)

            output, logits_variance = net(image, question, q_len)

            nll = -log_softmax(output)
            ud_ce_loss = (nll * y / 10).sum(dim=1).mean()

            gce_loss, variance_loss, undistorted_loss, variance_depressor = aleatoric_loss(logits_variance, output, y)
            aleatoric_uncertainty_loss = gce_loss + variance_loss + variance_depressor
            loss = ud_ce_loss + aleatoric_uncertainty_loss
            
            correct = output.detach().argmax(1) == answer
            running_loss += loss.item()

            batches_done += 1
            for c in correct:
                if c:
                    correct_counts += 1
                total_counts += 1

            pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct_counts / total_counts))

    val_acc = correct_counts / total_counts
    val_loss = running_loss / total_counts
    print('Validation Accuracy: {:.5f}'.format(val_acc))
    print('Validation Loss: {:.8f}'.format(val_loss))
    
#     dataset_object.close()
    return val_acc, val_loss

    
# aleatoric uncertainty
def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]



class AleatoricCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(AleatoricCrossEntropyLoss, self).__init__()
        self.elu = nn.ELU()
        self.categorical_crossentropy = undistorted_cross_entropy()
        # swd: 
        self.T = 100  # monte carlo number
        self.num_classes = 28
        

    def forward(self, logit_var, pred, true):
        std = torch.sqrt(logit_var)
        variance = logit_var

        variance_diff = torch.exp(variance) - torch.ones_like(variance)
        variance_depressor = torch.mean(variance_diff)
        undistorted_loss = self.categorical_crossentropy(pred, true)

        dist = torch.distributions.Normal(torch.zeros_like(std), std)

        monte_carlo_results = 0
        monte_carlo_results_gce = 0
        for i in range(0, self.T):
            mc_gce_loss, mc_gce_diff_loss = self.gaussian_categorical_crossentropy(pred, true, dist, undistorted_loss)
            monte_carlo_results = monte_carlo_results + mc_gce_diff_loss
            monte_carlo_results_gce = monte_carlo_results_gce + mc_gce_loss
        variance_loss = (monte_carlo_results / self.T)
        gce_loss = (monte_carlo_results_gce / self.T)

        return gce_loss, variance_loss, undistorted_loss, variance_depressor

    def gaussian_categorical_crossentropy(self, pred, true, dist, undistorted_loss):
        std_samples = (dist.sample_n(self.num_classes)).transpose(0, 1)
        std_samples = std_samples.view(std_samples.size(0), -1)

        distorted_loss = self.categorical_crossentropy(pred + std_samples, true)
        diff = undistorted_loss - distorted_loss

        return distorted_loss, -self.elu(diff)

    def categorical_cross_entropy(true, pred):
        return np.sum(true * np.log(pred), axis=1)


class undistorted_cross_entropy(nn.Module):
    def __init__(self):
        super(undistorted_cross_entropy, self).__init__()
        self.log_softmax = nn.LogSoftmax().cuda()

    def forward(self, pred, true):
        nll = -self.log_softmax(pred)
        return (nll * true / 10).sum(dim=1).mean()
    
    
    


if __name__ == '__main__':
    dataset_type = 'CLEVR'
    # input
    args = parse_args()
    decay = args.decay
    load_embd = args.load_embd
    out_name = args.out_name
    n_epoch = args.epoch
    
    out_directory = 'result/'+ out_name +'/'
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    print('Saving result to: ', out_directory)
    
    if not load_embd:
        with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
            dic = pickle.load(f)
        n_words = len(dic['word_dic']) + 1
        n_answers = len(dic['answer_dic'])
        print('Training word embeddings from scratch...')
    else:
        # add codes for loading GLOVE, embd dimensions, and out dim
        print('Loading GLOVE word embeddings...')
        pass
    
    
    # loading dataset using hdf5 imposing minimal overhead
    since = time.time()
    if dataset_type == "CLEVR":
        train_object = CLEVR('data/CLEVR_v1.0', transform=transform)
        val_object = CLEVR('data/CLEVR_v1.0', 'val', transform=None)
    else:
        train_object = GQA('data/gqa', transform=transform)
        val_object = GQA('data/gqa', 'val', transform=None)
    print('Dataset loaded in %.2f seconds' %(time.time()-since) )
    

    net = MACNetwork(n_words, dim_dict[dataset_type], classes=n_answers, max_step=4).to(device)
    net_running = MACNetwork(n_words, dim_dict[dataset_type], classes=n_answers, max_step=4).to(device)
    accumulate(net_running, net, 0)

    # criterion/loss defined in train/valid
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    # logging training information
    with open(out_directory + 'log.txt', 'w') as outfile:
        outfile.write('==== Training details ====\n')
        outfile.write('---- Model structure ----\n')
        outfile.write('Loading GLOVE embedding:  %s.     dictionary dim: %d. \n' %(load_embd, n_words))
        outfile.write('Hidden dimension: %d.     Output dimension: %d.\n' %(dim_dict[dataset_type], n_answers))
        
        outfile.write('\n---- Training detials ----\n')
        outfile.write('Batch size:  %d.     RA_decay: %f\n' %(batch_size, decay))
        outfile.write('Learning rate: %f.     Epochs: %d\n' %(learning_rate, n_epoch))
    
    
    learning_curve = np.zeros([0,4])
    acc_best = 0
    
    for epoch in range(n_epoch):
        train(epoch, dataset_type, train_object)
        train_acc, train_loss = valid(epoch, dataset_type, train_object)
        val_acc, val_loss = valid(epoch, dataset_type, val_object)
        
        # saving training result details.
        learning_curve = np.append(learning_curve, np.array([[train_acc,val_acc,train_loss,val_loss]]), axis = 0)
        savetxt(out_directory+'learn_curve.csv', learning_curve, delimiter=',')

        # saving trained models
        if val_acc > acc_best:
            with open(out_directory+'checkpoint.model', 'wb') as f:
    #         with open('checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(net_running.state_dict(), f)
            print('Accuracy increased from %.4f to %.4f, saved to %s. '%(acc_best, val_acc, out_directory+'checkpoint.model'))
            acc_best = val_acc
            
    print('The best validation accuracy: ', acc_best)
            
