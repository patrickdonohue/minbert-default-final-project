#import multitask_classifer.py
import os

lrs = ['0.8e-5', '1e-5', '1.2e-5']
dropout_rates = ['0.25', '0.3', '0.35']
directory = "hyperparameter_search"
epoch_nums = ['5', '10', '15']
option = 'finetune'
bert_hidden_sizes = ['768'],# '1536']#['512', '768', '1152']
max_tests = 10000

cnt = 0
for lr in lrs:
    for dr in dropout_rates:
        for epn in epoch_nums:
            for bhs in bert_hidden_sizes:
                if cnt < max_tests:
                    cmd = "python3 multitask_classifier.py --use_gpu --lr " + lr
                    cmd += " --hidden_dropout_prob " + dr
                    cmd += " --save_model_dir " + directory
                    cmd += " --epochs " + epn
                    cmd += " --option " + option
                    cmd += " --bert_hidden_size " + bhs
                    print()
                    print()
                    print('executing the following command:')
                    print(cmd)
                    os.system(cmd)
                    cnt += 1

    