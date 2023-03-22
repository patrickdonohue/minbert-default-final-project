import os
os.system("python3 multitask_classifier.py --use_gpu \
 --option pretrain --lr 1e-3 --epochs 3 --model_loader_filepath\
  SIMCSE_pretrained --task_list sts")

os.system("python3 multitask_classifier.py --use_gpu\
--option pretrain --lr 1e-3 --epochs 3 --model_loader_filepath\
SIMCSE_pretrained --task_list sts")

os.system("python3 multitask_classifier.py --use_gpu\
--option pretrain --lr 1e-3 --epochs 3 --model_loader_filepath\
SIMCSE_pretrained --task_list sts")