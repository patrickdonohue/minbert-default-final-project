import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, load_multitask_data, load_multitask_test_data, NLIDataset

from evaluation import test_model_multitask, model_eval_multitask, model_eval_test_multitask #,model_eval_sst


TQDM_DISABLE=False #change?

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


# class simcseSupervisedLoss(nn.Module):
#     def __init__(self):
#         super(simcseSupervisedLoss, self).__init__()

#     def forward(self, cos_sim12, cos_sim23, cos_sim31, tau):
#         # print('shapes')
#         # print(cos_sim12.shape)
#         # loggednum = torch.logsumexp(cos_sim12 / tau)
#         # loggeddenom = torch.logsumexp(cos_sim / tau)
#         num = torch.sum(torch.exp(cos_sim12 / tau))
#         denom = torch.sum(torch.exp(cos_sim12/tau) + torch.exp(cos_sim31 / tau))
#         return -torch.log(num/denom)

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.sent_classifier = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES )
        self.paraphrase_classifier = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.sim_classifier = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 1)

        ### for constrastive	
        self.margin = 0.05 #hard code fpr now	
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.simcse_classifier = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        
        outputs = self.bert(input_ids, attention_mask)
        pooler_output = outputs['pooler_output']
        # pooler_output = self.dropout(pooler_output)
        # logits = self.classifier(pooler_output)
        return pooler_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        fwd = self.forward(input_ids, attention_mask)
        return self.sent_classifier(fwd)
        


    def predict_paraphrase(self,input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        para1_embed = self.forward(input_ids_1, attention_mask_1)
        para2_embed = self.forward(input_ids_2, attention_mask_2)
        paraphrase_embeddings = torch.cat((para1_embed, para2_embed), dim=1) #is this broken?
        paraphrase_logits = self.paraphrase_classifier(paraphrase_embeddings)
        return paraphrase_logits


    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        sim1_embed = self.forward(input_ids_1, attention_mask_1)
        sim2_embed = self.forward(input_ids_2, attention_mask_2)
        sim_embeddings = torch.cat((sim1_embed, sim2_embed), dim=1)
        #sim_logits = self.paraphrase_classifier(sim_embeddings)
        sim_logits = self.sim_classifier(sim_embeddings)
        return sim_logits

    def predict_nli(self, batch, device, args):#, lossFn):
        '''Given a batch of triplets of sentences, outputs a loss directly. 
            Doesn't need to predict diddly squat, because we don't have labels.
            Also this isn't a task in and of itself, it's just a training method.
        '''
        # (token_ids1, token_type_ids1, attention_mask1,
        # token_ids2, token_type_ids2, attention_mask2,
        # token_ids3, token_type_ids3, attention_mask3) = batch
        token_ids1 = batch['token_ids1'].to(device)
        token_type_ids1 = batch['token_type_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        token_ids2 = batch['token_ids2'].to(device)
        token_type_ids2 = batch['token_type_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        token_ids3 = batch['token_ids3'].to(device)
        token_type_ids3 = batch['token_type_ids3'].to(device)
        attention_mask3 = batch['attention_mask3'].to(device)
        
        # self.simcse_classifier()
        embed1 = self.simcse_classifier(self.forward(token_ids1, attention_mask1))
        embed2 = self.simcse_classifier(self.forward(token_ids2, attention_mask2))
        embed3 = self.simcse_classifier(self.forward(token_ids3, attention_mask3))

        cos_sim12 = self.cosine_similarity(embed1, embed2)
        cos_sim23 = self.cosine_similarity(embed2, embed3)
        cos_sim13 = self.cosine_similarity(embed1, embed3)
        #cos_sim31 = self.cosine_similarity(embed3, embed1)
        #assert(torch.all(torch.eq(cos_sim13, cos_sim31)))
        #lossFn = simcseSupervisedLoss()
        #loss = lossFn.forward(cos_sim12, cos_sim23, cos_sim31, args.tau)

        tau = args.tau
        num = torch.exp(cos_sim12 / tau)
        denom = torch.sum(torch.exp(cos_sim12/tau) + torch.exp(cos_sim13 / tau))
        loss = -torch.log(num/denom)
        #loss = torch.ones(len(cos_sim12)).to(device)
        lossfn = torch.nn.L1Loss()
        zeros = torch.zeros(len(cos_sim12)).to(device)
        loss = lossfn(loss, zeros) / args.batch_size #, reduction = 'sum')#is this last division OK?
        return loss

    # def simcseUnsupervised(self,batch,device,args):
    '''I can't be asked to do this.'''
    #     token_ids = batch['token_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)


    



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    #clobbering filepath!!
    #filepath = "temp.pt"
    filepath = 'models/' + filepath
    torch.save(save_info, filepath)
    print(f"saved the model to {filepath}")
    # print("WE AREN'T SAVING MODELS THESE DAYS - TOO MUCH SPACE")



## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data 
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data, nli_train_data = load_multitask_data(args.nli_train, args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data, nli_dev_data = load_multitask_data(args.nli_dev, args.sst_dev,args.para_dev,args.sts_dev, split ='train')
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    nli_train_data = NLIDataset(nli_train_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    nli_train_dataloader = DataLoader(nli_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn = nli_train_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn = para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn = para_train_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn = sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn = sts_dev_data.collate_fn)
    
    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    if args.option == 'pretrain' and args.model_loader_filepath != "BERT":
        print('using pretrained weights from ' + args.model_loader_filepath)
        saved = torch.load("models/" + args.model_loader_filepath)
        config = saved['model_config']
        model.load_state_dict(saved['model'])
    model = model.to(device)
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    if args.optimizer == 'SGD':
        print('USING SGD OPTIMIZER')
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    best_dev_acc = 0
    stats = {}
    best_overall_score = 0.507
    best_stats = {'overall_score': 0}

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0 
        numBatches = args.batch_iters #should be 750
        dataloaders = [sst_train_dataloader, nli_train_dataloader, para_train_dataloader, sts_train_dataloader]
        task_names = ['sst', 'nli', 'para', 'sts']
        dataloadersDict = {'sst': iter(sst_train_dataloader), 'nli': iter(nli_train_dataloader),
                            'para': iter(para_train_dataloader), 'sts': iter(sts_train_dataloader)}

        #lossFnsDict = {'sst': F.cross_entropy, 'nli': torch.nn.L1Loss, 'para': torch.nn.BCELoss, 'sts':F.mse_loss}
        for batchI in tqdm(range(numBatches), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            for task_name in task_names:
                batch = next(dataloadersDict[task_name])
                #lossFn = lossFnsDict[task_name]
                optimizer.zero_grad()
                if task_name == "sst":
                    if args.just_simcse == 'T':
                        continue
                    b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])
                    b_ids = b_ids.to(device)
                    b_mask = b_mask.to(device)
                    b_labels = b_labels.to(device)
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                elif task_name == "nli":
                    if args.use_simcse == 'F' or args.option == 'pretrain':
                        continue 
                    loss = model.predict_nli(batch, device, args)
                elif task_name == 'para':
                    if args.just_simcse == 'T':
                        continue
                    tids1 = batch['token_ids_1'].to(device)
                    mask1 = batch['attention_mask_1'].to(device)
                    tids2 = batch['token_ids_2'].to(device)
                    mask2 = batch['attention_mask_2'].to(device)
                    b_labels = batch['labels'].to(device)
                    b_labels = b_labels.to(torch.float32)
                    logits = model.predict_paraphrase(tids1, mask1, tids2, mask2)
                    lossFn = torch.nn.BCELoss(reduction='sum')
                    sig = nn.Sigmoid()
                    loss = lossFn(sig(logits.view(-1)), b_labels.view(-1))
                elif task_name == 'sts':
                    if args.just_simcse == 'T':
                        continue
                    tids1 = batch['token_ids_1'].to(device)
                    mask1 = batch['attention_mask_1'].to(device)
                    tids2 = batch['token_ids_2'].to(device)
                    mask2 = batch['attention_mask_2'].to(device)
                    b_labels = batch['labels'].to(device)
                    b_labels = b_labels.to(torch.float32)
                    logits = model.predict_similarity(tids1, mask1, tids2, mask2)
                    loss = F.mse_loss(logits.view(-1), b_labels.view(-1), reduction='sum') / args.batch_size
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / (num_batches)
        # doubly sample paraphrase lr? higher tau should handle this.

        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        # (paraphrase_accuracy, para_y_pred, para_sent_ids,
        #         sentiment_accuracy,sst_y_pred, sst_sent_ids,
        #         sts_corr, sts_y_pred, sts_sent_ids) = \
        stats = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        if stats['overall_score'] > best_overall_score:
            print("NEW BEST SCORE!!!")
            best_overall_score = stats['overall_score']
            save_model(model, optimizer, args, config, 'BEST_MODEL.pt')
        if stats['overall_score'] > best_stats['overall_score']:
            best_stats = stats

        #print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    if args.save_model == 'T':
        save_model(model, optimizer, args, config, args.filepath)

    # writePredictions = args.write_predictions == 'T'
    # if writePredictions:
    #     stats = test_model_multitask(args, model, device, evalOnTest = writePredictions, writePreds = writePredictions)
    return model, best_stats


# def test_model(args, model):
#     '''We do not use this right now!!'''
#     with torch.no_grad():
#         device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
#         saved = torch.load(args.filepath)
#         config = saved['model_config']

#         model = MultitaskBERT(config)
#         model.load_state_dict(saved['model'])
#         model = model.to(device)
#         print(f"Loaded model to test from {args.filepath}")
#         return test_model_multitask(args, model, device)

def writePredictions(args):
    '''Just write predictions, do not train a model at all'''
    # print(args.lr)
    # print(args.prefix)
    # exit(0)
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.model_loader_filepath)
        config = saved['model_config']
        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        return test_model_multitask(args, model, device, evalOnTest = False, writePreds = True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")



    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="finetune")

    # do not touch these 7
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--sst_dev_out", type=str, default="sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="sst-test-output.csv")
    parser.add_argument("--para_dev_out", type=str, default="para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="para-test-output.csv")
    parser.add_argument("--sts_dev_out", type=str, default="sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--save_model_dir", type=str, default = "models")

    ## for SimCSE
    parser.add_argument('--nli_train', type=str, default='data/nli_for_simcse-train.csv')
    parser.add_argument('--nli_dev', type=str, default='data/nli_for_simcse-dev.csv')
    parser.add_argument('--nli_test', type=str, default='data/nli_for_simcse-test.csv')
    parser.add_argument('--tau', type=float, default=5e-2) #TODO. Does this default value make any sense?
    parser.add_argument('--batch_iters', type=int, default=700) 
    parser.add_argument('--use_simcse', type=str, default='T')
    parser.add_argument('--just_simcse', type=str, default='F')
    parser.add_argument('--save_model', type=str, default='F')
    parser.add_argument('--write_predictions', type=str, default='F')
    parser.add_argument('--write_predictions_only', type=str, default='F')
    parser.add_argument('--model_loader_filepath', type=str, default='models/BEST_MODEL.pt')
    parser.add_argument('--optimizer', type=str, default='ADAMW')
    #parser.add_argument('--load_previous_model_for_finetuning', type=str default='')


    args = parser.parse_args()
    return args

NAME = 'NEWEST_model_summaries.csv'
STATS = ['lr', 'option', 'dropout_prob', 'epochs', 'just_simcse', 'use_simcse', 
         'tau', 'paraphrase_detection_acccuracy', 'sentiment_classification_accuracy',
          'semantic_textual_similarity_correlation', 'overall_score']
def createDataframe():
    print("CREATING DATAFRAME, CLOBBERING OLD THINGS")
    df = pd.DataFrame(data = [], columns = STATS)
    df.to_csv(NAME, index = False)
    return

def saveStats(newData):
    # d = {}
    # for i in range(len(STATS)):
    #     d[STATS[i]] = newData[i]
    df = pd.read_csv(NAME)
    df = df.append(newData, ignore_index = True)
    df.to_csv(NAME, index = False)
    return

def stripBadChars(s):
    s = s.replace('{', '')
    s = s.replace('}', '')
    s = s.replace(':', '')
    s = s.replace(',', '-')
    s = s.replace(' ', '_')
    s = s.replace('\'', '')
    s = s.replace('\"', '')
    return s

def areTF(strs):
    return all([s in ['T', 'F'] for s in strs])

def writeSomethingStupid():
    with open('predictions/newfile.txt', "x") as f:
        f.write('lolz')

if __name__ == "__main__":
    args = get_args()
    #args.filepath = f'{args.save_model_dir}/{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    #args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    #args.filepath = 'temp.pt'
    if not areTF([args.use_simcse, args.just_simcse, args.save_model, args.write_predictions,
                  args.write_predictions_only]):
        raise Exception('bad arg format')
    if args.write_predictions_only == 'T':
        args.prefix = 'BEST_MODEL'
        writePredictions(args)
        exit(0)
    if args.option == "pretrain":
        print('make sure you reset hyperparams! Lr should be 1e-3, epochs prob just 1-2')
        if args.model_loader_filepath == 'BERT':
            print('using pretrained bert embeddings (instead of already used simcse model')


    if args.use_simcse == 'F' and args.just_simcse == 'T':
        raise Exception('bad arg format')
    hyperparams = {'lr': args.lr, 'option': args.option, 'epochs':args.epochs, 
                    'dropout_prob': args.hidden_dropout_prob, 'just_simcse': args.just_simcse,
                    'use_simcse': args.use_simcse, 'tau': args.tau}
    seed_everything(args.seed)  # fix the seed for reproducibility
    args.prefix = stripBadChars(str(hyperparams))
    args.filepath = 'model_' + args.prefix + '.pt'
    model, stats = train_multitask(args)
    print('saving stats')
    hyperparams.update(stats)
    saveStats(hyperparams)
    
    # TODO - write a method to use pretrained Simcse embeddings, and just finetune heads for the three tasks

