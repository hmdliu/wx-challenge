import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id

MODEL_PATH_LIST = [
    './save/0528_sa1_02/model_epoch_3_mean_f1_0.6441.bin',
    './save/0528_sa1_03/model_epoch_2_mean_f1_0.6327.bin',
    './save/0528_sa1_04/model_epoch_3_mean_f1_0.6321.bin',
    './save/0528_sa2_02/model_epoch_3_mean_f1_0.6322.bin',
    './save/0528_sa3_01/model_epoch_4_mean_f1_0.6313.bin',
    './save/0528_sa3_03/model_epoch_4_mean_f1_0.6301.bin',
    './save/0528_sa3_04/model_epoch_8_mean_f1_0.6410.bin',
    './save/0528_sa4_02/model_epoch_3_mean_f1_0.6325.bin',
    './save/0528_sa6_02/model_epoch_3_mean_f1_0.6357.bin',
    './save/0528_sa6_04/model_epoch_5_mean_f1_0.6358.bin',
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
