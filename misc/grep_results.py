
import os

collect = []
keyword_list = ['0528']
exp_log_list = os.listdir('./logs/')

for log in exp_log_list:
    if any([(log.find(kw) != -1) for kw in keyword_list]):
        with open(os.path.join('./logs/', log), 'r') as f:
            best_pred_line = f.read().split('\n')[-2]
            if best_pred_line.find('Best Pred') != -1:
                collect.append(f"{os.path.join('./logs/', log)}: {best_pred_line.split()[-1]}")

print('\n'.join(sorted(collect)))
