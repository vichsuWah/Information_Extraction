from dataset import Cinnamon_Dataset, DataLoader, tags
from train import BertTokenizer, BertJapaneseTokenizer, Model, pretrained_weights, train

import os, warnings, argparse
warnings.filterwarnings('ignore')


def parse_args(string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-5,
                        type=float, help='leanring rate')
    parser.add_argument('--epoch', default=50,
                        type=int, help='epochs')
    parser.add_argument('--batch-size', default=4,
                        type=int, help='batch size')
    parser.add_argument('--gpu', default="0",
                        type=str, help="0:1080ti 1:1070")
    parser.add_argument('--num-workers', default=8,
                        type=int, help='dataloader num workers')
    parser.add_argument('--model', default='naive',
                        type=str, help='naive,blstm')
    parser.add_argument('--delta', default=11,
                        type=int, help='dataset delta cat together')
    parser.add_argument('--decline-lr', action='store_true',
                                  help='decline lr if valid not better')
    parser.add_argument('--cinnamon-data-path', default='/media/D/ADL2020-SPRING/project/cinnamon/',
                        type=str, help='cinnamon dataset')
    parser.add_argument('--load-model', default='ckpt/epoch_6_model_loss_0.4579.pt',
                        type=str, help='.pt model file ')
    parser.add_argument('--save-path', default='ckpt/',
                        type=str, help='.pt model file save dir')
    
    args = parser.parse_args() if string is None else parser.parse_args(string)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    return args

if __name__=='__main__':
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    ## load tokenizer
    #tokenizer = BertTokenizer.from_pretrained(pretrained_weights)#, do_lower_case=True)
    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_weights)#, do_lower_case=True)

    ## load dataset
    train_dataset = Cinnamon_Dataset(f'{args.cinnamon_data_path}/train/', tokenizer, args.delta)
    valid_dataset = Cinnamon_Dataset(f'{args.cinnamon_data_path}/dev/', tokenizer, args.delta)

    train_dataloader = DataLoader(train_dataset,
                                 batch_size = args.batch_size,
                                 num_workers = args.num_workers,
                                 collate_fn = train_dataset.collate_fn,
                                 shuffle = True)
    valid_dataloader = DataLoader(valid_dataset,
                                 batch_size = args.batch_size*4,
                                 num_workers = args.num_workers,
                                 collate_fn = valid_dataset.collate_fn)
    
    ## train
    train(args, train_dataloader, valid_dataloader)
    
