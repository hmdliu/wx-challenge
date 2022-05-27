
import os

exp_dir_list = os.listdir('./save/')

for dir in exp_dir_list:
    ckpt_list = os.listdir(os.path.join('./save/', dir))
    if len(ckpt_list) > 1:
        ckpt_dict = {int(ckpt.split('_')[2]):os.path.join('./save/', dir, ckpt) for ckpt in ckpt_list}
        for ep, path in ckpt_dict.items():
            if ep != max(ckpt_dict.keys()):
                print(f'Removing {path}')
                os.remove(path)
