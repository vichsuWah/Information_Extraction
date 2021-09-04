from dataset import tags, Cinnamon_Dataset_Testing, DataLoader
from train import *
from utils.convert import *
from utils.score import * 

from collections import Counter 
import torch.nn.functional as F
import os, warnings, argparse
warnings.filterwarnings('ignore')

def parse_args(string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1,
                        type=int, help='batch size')
    parser.add_argument('--gpu', default="1",
                        type=str, help="0:1080ti 1:1070")
    parser.add_argument('--num-workers', default=8,
                        type=int, help='dataloader num workers')
    parser.add_argument('--model', default='naive',
                        type=str, help='naive,blstm')
    parser.add_argument('--delta', default=11,
                        type=int, help='data delta cat together')
    parser.add_argument('--postprocess', action='store_true',
                                  help='do postprocessing ?')
    parser.add_argument('--cinnamon-data-path', default='/media/D/ADL2020-SPRING/project/cinnamon/',
                        type=str, help='cinnamon dataset')
    parser.add_argument('--dev_or_test', default='dev',
                        type=str, help='dev or test set')
    parser.add_argument('--load-model', default='./naive_baseline/ckpt/epoch_34.pt',
                        type=str, help='.pt model file ')
    parser.add_argument('--save-result-path', default='./naive_baseline/result/',
                        type=str, help='.pt model file save dir')
    parser.add_argument('--ref-file', default='/media/D/ADL2020-SPRING/project/cinnamon/dev/dev_ref.csv',
                        type=str, help='calcu score ref file')    
    parser.add_argument('--score', action='store_true')
    
    args = parser.parse_args() if string is None else parser.parse_args(string)
    if not os.path.exists(args.save_result_path): os.makedirs(args.save_result_path)
    return args


def fuller(text):
    candidate = (ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7'),ord('8'),ord('9'),
            ord('('),ord(')'),ord('~'),)
    text_out = ''
    for c in text:
        if ord(c) in candidate:
            text_out += chr(ord(c)+65248)
        else:
            text_out += c
    return text_out

def post_process(value, tag, text):
    
    if tag=='質問箇所TEL/FAX':
        return text.replace('イ．','').replace('ア．','').replace('．','').replace(' ','') 
    elif tag=='質問箇所所属/担当者':
        return text.replace('イ．','').replace('ア．','').replace('．','').replace(' ','') 
    elif tag in ['資格申請送付先',
                 '資格申請送付先部署/担当者名',
                 '入札書送付先',
                 '入札書送付先部署/担当者名',]:
        return text.replace('イ．','').replace('ア．','').replace('．','').replace(' ','') 
 

    '''
        value = value.replace('##l:','ＴＥＬ：').replace('tel:','ＴＥＬ：').replace('Tel:','ＴＥＬ：').replace('TEL:','ＴＥＬ：')
        value = value.replace('fax:','ＦＡＸ：').replace('Fax:','ＦＡＸ：').replace('FAX:','ＦＡＸ：')
    
    print(text)
    input("")
    
    # 半形 轉 全形
    value = fuller(value)
    '''
    value_ret = ''
    for c in value:
        if c in text:
            c = c
        elif chr(ord(c)+65248) in text:
            c = chr(ord(c)+65248)
        elif ord('a')<=ord(c) and ord(c)<=ord('z'): #小寫轉大寫
            if chr(ord(c)-32) in text: #小寫轉大寫 半形
                c = chr(ord(c)-32)
            elif chr(ord(c)+65248-32) in text: #小寫轉大寫 + 轉全形
                c = chr(ord(c)+65248-32)
        elif ord('A')<=ord(c) and ord(c)<=ord('Z'): #大寫轉小寫
            if chr(ord(c)+32) in text: #大寫轉小寫 半形
                c = chr(ord(c)+65248+32)            
            elif chr(ord(c)+65248+32) in text: #大寫轉小寫 + 轉全形
                c = chr(ord(c)+65248+32)
        else:
            pass
            #print(c, text, value)
        value_ret += c 
    return value_ret
    
def inference(args, tokenizer, dataloader):
   
    if args.model == 'naive':
        model = Model()
    elif args.model == 'blstm':
        model = Model_BLSTM()
    model.load_state_dict(torch.load(args.load_model)['state_dict']) #.cuda().eval()
    model.eval()
    
    with torch.no_grad():
        total_dataframe = None 
        for iii,(_input, _label, token_indexs, doc_id, sample) in enumerate(dataloader):
            sample['Prediction'] = ""
            sample['Tag'] = ""
            sample['Value'] = ""

            _output = model(_input)[0]
            prob = F.sigmoid(_output)

            for i,tag in enumerate(tags):
                index = []
                values = []
                for j in range(prob.size(0)):
                    if prob[j,i] > 0.5:
                        values.append(_input[0][j])
                        index.append(token_indexs[0][j])

                if len(values)>0:
                    index = Counter(index).most_common()[0][0]
                    
                    value_str = tokenizer.decode(values, skip_special_tokens=True).replace(" ","")
                    if args.postprocess:
                        value_str = post_process(value_str, tag, sample.loc[sample['Index']==index, 'Text'].item())
                    else:
                        value_str = value_str 

                    # add a tag&value to <Tag> <Value>
                    if sample[sample['Index']==index]['Tag'].item() == "":
                        sample.loc[sample['Index']==index, 'Tag'] = "{}".format(tag)
                        sample.loc[sample['Index']==index, 'Value'] = "{}".format(value_str)
                    else:
                        sample.loc[sample['Index']==index, 'Tag'] += ";{}".format(tag)
                        sample.loc[sample['Index']==index, 'Value'] += ";{}".format(value_str)

            total_dataframe = total_dataframe.append(sample) if isinstance(total_dataframe, pd.DataFrame) else sample

            print(f'\t[Info] [{iii+1}/{len(dataloader)}]', end='   \r')
            
        print('\t[Info] finish inference ')
            
        # sort dataframe & save results
        total_dataframe = total_dataframe.sort_values(by=['id','Index'], ascending=[True,True])
        total_dataframe = total_dataframe.drop('Page No', axis=1).drop('Parent Index', axis=1).drop('Is Title', axis=1).drop(
                        'Is Table', axis=1).drop('id', axis=1).drop('Index', axis=1)
        total_dataframe.to_csv(f'{args.save_result_path}/result.csv', encoding='utf8')
        
        # convert results to submission format
        convert(f'{args.save_result_path}/result.csv', f'{args.save_result_path}/submission.csv')
        
        # score
        if args.score:
            s = score(args.ref_file, f'{args.save_result_path}/submission.csv')
            print('\t[Info] score:', s)
        

if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_weights)#, do_lower_case=True)

    dataset = Cinnamon_Dataset_Testing(f'/media/D/ADL2020-SPRING/project/cinnamon/{args.dev_or_test}/', tokenizer, args.delta)
    dataloader = DataLoader(dataset,
                             batch_size=1,
                             collate_fn=dataset.collate_fn,
                             shuffle=False)

    inference(args, tokenizer, dataloader)
    
