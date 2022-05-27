import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id

MODEL_PATH_LIST = [
    'save/0526_filter_01/model_epoch_3_mean_f1_0.6199.bin',
    'save/0526_filter_02/model_epoch_4_mean_f1_0.6206.bin',
    'save/0526_filter_03/model_epoch_3_mean_f1_0.6259.bin',
    'save/0526_filter_04/model_epoch_4_mean_f1_0.6181.bin',
    'save/0526_filter_05/model_epoch_7_mean_f1_0.6216.bin',
    'save/0526_filter_07/model_epoch_7_mean_f1_0.6223.bin',
    'save/0526_filter_08/model_epoch_5_mean_f1_0.6171.bin',
    'save/0526_filter_09/model_epoch_6_mean_f1_0.6228.bin',
    'save/0526_filter_10/model_epoch_4_mean_f1_0.6240.bin',
    'save/0526_filter_11/model_epoch_8_mean_f1_0.6168.bin',
    'save/0526_filter_12/model_epoch_3_mean_f1_0.6266.bin',
    'save/0526_filter_14/model_epoch_8_mean_f1_0.6191.bin',
    'save/0526_filter_15/model_epoch_4_mean_f1_0.6222.bin',
    'save/0526_filter_17/model_epoch_7_mean_f1_0.6246.bin',
    'save/0526_filter_19/model_epoch_7_mean_f1_0.6206.bin',
    'save/0526_filter_20/model_epoch_8_mean_f1_0.6258.bin',
]

def ensemble_inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model_list = []
    for i in range(len(MODEL_PATH_LIST)):
        checkpoint = torch.load(MODEL_PATH_LIST[i], map_location='cpu')
        model = checkpoint['model_class'](checkpoint['args'])
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        model_list.append(model)

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pred_list = []

            # sum
            for i in range(len(model_list)):
                pred_list.append(model_list[i](batch, inference=True))
            pred = sum(pred_list)

            # # max
            # for i in range(len(model_list)):
            #     pred_list.append(model_list[i](batch, inference=True).view(-1, 1, 200))
            # pred = torch.cat(pred_list, dim=1).max(dim=1).values
            
            predictions.extend(torch.argmax(pred, dim=1).cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    ensemble_inference()
