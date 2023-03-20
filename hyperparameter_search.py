#import multitask_classifer.py
import os

lrs = ['0.8e-5', '1e-5', '1.2e-5']
#dropout_rates = ['0.25', '0.3', '0.35']
dropout_rates = ['0.3']
directory = "models"
epoch_nums = ['5', '10', '20', '50']
option = 'finetune'
bert_hidden_sizes = ['768'] # '1536']#['512', '768', '1152']
taus = ['5e-2', '1e-1'] #look this up!!
batch_iters = '750' #can't be more than 755
max_tests = 5 #fix!!!


cnt = 0
for epn in epoch_nums:
    for dr in dropout_rates:
        for lr in lrs:
            for bhs in bert_hidden_sizes:
                for tau in taus:
                    for use_sim in ['True', 'False']:
                        if cnt < max_tests:
                            cmd = "python3 multitask_classifier.py --use_gpu --lr " + lr
                            cmd += " --hidden_dropout_prob " + dr
                            cmd += " --save_model_dir " + directory
                            cmd += " --epochs " + epn
                            cmd += " --option " + option
                            #cmd += " --bert_hidden_size " + bhs
                            cmd += " --tau " + tau
                            cmd += " --batch_iters " + batch_iters
                            cmd += " --use_simcse " + use_sim
                            print('executing the following command:')
                            print(cmd)
                            os.system(cmd)
                            cnt += 1

    