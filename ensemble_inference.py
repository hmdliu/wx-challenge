import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal

MODEL_PATH_LIST = [
    './save/base_1/model_epoch_5_mean_f1_0.5776.bin',
    './save/base_2/model_epoch_3_mean_f1_0.5781.bin',
    './save/base_3/model_epoch_4_mean_f1_0.5744.bin',
    './save/base_4/model_epoch_5_mean_f1_0.5876.bin',
    './save/base_5/model_epoch_4_mean_f1_0.5772.bin',
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
    model_list = [MultiModal(args) for i in range(len(MODEL_PATH_LIST))]
    for i in range(len(MODEL_PATH_LIST)):
        checkpoint = torch.load(MODEL_PATH_LIST[i], map_location='cpu')
        model_list[i].load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model_list[i] = torch.nn.parallel.DataParallel(model_list[i].cuda())
        model_list[i].eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
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
