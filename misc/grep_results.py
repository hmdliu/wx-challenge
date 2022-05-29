
import os
import sys

collect = []
keyword = sys.argv[1]
exp_log_list = os.listdir('./logs/')

for log in exp_log_list:
    if log.find(keyword) != -1:
        with open(os.path.join('./logs/', log), 'r') as f:
            best_pred_line = f.read().split('\n')[-2]
            if best_pred_line.find('Best Pred') != -1:
                collect.append(f"{os.path.join('./logs/', log)}: {best_pred_line.split()[-1]}")

print('\n'.join(sorted(collect)))
