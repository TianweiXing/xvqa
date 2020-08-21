"""
Author: Tianwei Xing (twxing@ucla.edu)
"""

import torch 
import torch.nn as nn

import pickle
import time
# from torch import optim  # needless

# from model_gqa import MACNetwork
from src.model import MACNetwork
from src.model_predu import MACNetwork as MACNetwork_predu

from src.dataset import CLEVR, collate_data, transform, GQA
from torch.utils.data import DataLoader
import multiprocessing

from tqdm.notebook import tqdm


from src.edl_utils import relu_evidence



class Load_model(nn.Module):
    
    def __init__(self, model_path, model_type):
        super(Load_model, self).__init__()
#         batch_size = 128
#         n_epoch = 25
        dim_dict = {'CLEVR': 512,
                    'gqa': 2048}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_type = model_type  # one of [MAC, EDL, PredU]
        
        dataset_type = 'CLEVR'
        with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
            dic = pickle.load(f)
        self.n_words = len(dic['word_dic']) + 1
        self.n_answers = len(dic['answer_dic'])
        
        if self.model_type == 'PredU':
            self.model_load = MACNetwork_predu(self.n_words, dim_dict[dataset_type], classes=self.n_answers, max_step=4)
        else:
            self.model_load = MACNetwork(self.n_words, dim_dict[dataset_type], classes=self.n_answers, max_step=4)
        self.model_load = self.model_load.to(self.device)
        # optimizer = optim.Adam(model_load.parameters())  # needless


        checkpoint = torch.load(model_path)
        self.model_load.load_state_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint) # needless
        self.model_load.eval()
        
        # model evaluation result
        self.out_prediction = []
        self.out_correct = []
        self.out_uncertainty = []

        
    def evaluate(self):
        print('Evaluating loaded model on validation dataset.....')
        since = time.time()
        
#         def valid(epoch, dataset_type):
        batch_size = 128
        dataset_object = CLEVR('data/CLEVR_v1.0', 'val', transform=None)
        valid_set = DataLoader(
            dataset_object, batch_size=4*batch_size, num_workers=1, collate_fn=collate_data
#             dataset_object, batch_size=4*batch_size, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
        )
        dataset = iter(valid_set)
        print('Data loader created.')


        criterion = nn.CrossEntropyLoss()
        epoch = 0
        
        # output prediction result and uncertainty
        self.out_prediction = []
        self.out_correct = []
        self.out_uncertainty = []
        
        self.model_load.train(False)
        correct_counts = 0
        total_counts = 0
        running_loss = 0.0
        batches_done = 0
        with torch.no_grad():
            pbar = tqdm(dataset)
            for image, question, q_len, answer in pbar:
                image, question, answer = (
                    image.to(self.device),
                    question.to(self.device),
                    answer.to(self.device),
                )
                if self.model_type == 'PredU':
                    output, variance = self.model_load(image, question, q_len)
                    pred_answer = output.detach().argmax(1) 
                    self.out_prediction.append(pred_answer)
                    self.out_uncertainty.append(variance)
                elif self.model_type == 'EDL': 
                    output = self.model_load(image, question, q_len)
                    pred_answer = output.detach().argmax(1) 
                    self.out_prediction.append(pred_answer)
                    
                    num_classes = self.n_answers
                    evidence = relu_evidence(output)
                    alpha = evidence + 1
                    uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                    self.out_uncertainty.append(uncertainty)
                    
                else: 
                    output = self.model_load(image, question, q_len)
                    pred_answer = output.detach().argmax(1) 
                    self.out_prediction.append(pred_answer)
                    
                loss = criterion(output, answer)
                
                
                correct = pred_answer == answer
                self.out_correct.append(correct)
                running_loss += loss.item()

                batches_done += 1

                correct_counts = correct_counts + correct.sum().item()
                total_counts = total_counts + list(correct.size())[0]

                pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct_counts / total_counts))

        print('Validation Accuracy: {:.5f}'.format(correct_counts / total_counts))
        print('Validation Loss: {:.8f}'.format(running_loss / total_counts))
        print('Evaluation time: %.2f seconds' %(time.time()- since))

        dataset_object.close()
        
#         if out_resut:
#             return torch.cat(out_prediction, dim=0) , torch.cat(out_uncertainty, dim=0) 
        
        

        
    def predict(self, image, question, q_len):
#         print('Predicting.....')
       
        image, question = (
                    image.to(self.device),
                    question.to(self.device),
#                     answer.to(self.device),
                )
                
        if self.model_type == 'MAC':
            output = self.model_load(image, question, q_len)
            return output.cpu().detach().numpy()
        
        elif self.model_type == 'EDL':
            output = self.model_load(image, question, q_len)
            
            num_classes = self.n_answers
#             output = model_load(img_variable, que_variable, q_len_var)
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
#             _, preds = torch.max(output, 1)
#             prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
#             output = output.flatten()
#             prob = prob.flatten()
#             preds = preds.flatten()
#             print("Predict:", preds[0])
#             print("Probs:", prob)
#             print("Uncertainty:", uncertainty)

            return output.cpu().detach().numpy(), uncertainty.cpu().detach().numpy()
        
        
        elif self.model_type == 'PredU':
            output, logits_variance= self.model_load(image, question, q_len)
            return output.cpu().detach().numpy(), logits_variance.cpu().detach().numpy()
        
        else:
            print('Wrong Model Type!')
            return
            
        