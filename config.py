import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--final_dropout', type=float, default=0.0, help='final dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='data/annotations/labeled_preprocessed.json')
    parser.add_argument('--test_annotation', type=str, default='data/annotations/test_a_preprocessed.json')
    parser.add_argument('--train_zip_feats', type=str, default='data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='data/zip_feats/test_a.zip')
    parser.add_argument('--test_output_csv', type=str, default='pred/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=64, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
    parser.add_argument('--sample_anns', default=False, type=bool, help="subsampling class 13")
    parser.add_argument('--preprocess_text', default=True, type=bool, help="remove stop words")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save/v1')
    parser.add_argument('--ckpt_file', type=str, default='save/v1/model.bin')
    parser.add_argument('--export_bound', default=0.63, type=float, help='save checkpoint if mean_f1 > export_bound')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=12, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=500, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--es_patience', type=int, default=3, help='Early stop patience')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='Learning rate scheduler')

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-macbert-base')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=128)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=768, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")
    parser.add_argument('--simam', type=bool, default=False, help="SimAM before nextvlad")
    parser.add_argument('--co_att', type=bool, default=True, help="vision & text co-attention")
    parser.add_argument('--co_att_head', type=int, default=1, help="number of attention heads")


    return parser.parse_args()
