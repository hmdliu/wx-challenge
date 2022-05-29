
import os

collect = []
keyword_list = ['filter']
exp_dir_list = os.listdir('./save/')

for dir in exp_dir_list:
    ckpt_list = os.listdir(os.path.join('./save/', dir))
    if len(ckpt_list) > 0 and any([(dir.find(kw) != -1) for kw in keyword_list]):
        ckpt_dict = {int(ckpt.split('_')[2]):os.path.join('./save/', dir, ckpt) for ckpt in ckpt_list}
        collect.append(f"'{ckpt_dict[max(ckpt_dict.keys())]}',")

print('\n'.join(sorted(collect)))
