from dataset import tags, Cinnamon_Dataset_Testing_v2, DataLoader
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
    parser.add_argument('--postprocess', action='store_true',
                                  help='do postprocessing ?')
    parser.add_argument('--cinnamon-data-path', default='/media/D/ADL2020-SPRING/project/cinnamon/',
                        type=str, help='cinnamon dataset')
    parser.add_argument('--dev_or_test', default='dev',
                        type=str, help='dev or test set')
    parser.add_argument('--load-model', default='./naive_baseline/ckpt/epoch_34.pt',
                        type=str, help='.pt model file ')
    parser.add_argument('--save-result-path', default='./v2/result/',
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
    
    if tag=='????????????TEL/FAX':
        return text.replace('??????','').replace('??????','').replace('???','').replace(' ','') 
    elif tag=='??????????????????/?????????':
        return text.replace('??????','').replace('??????','').replace('???','').replace(' ','') 
    elif tag in ['?????????????????????',
                 '???????????????????????????/????????????',
                 '??????????????????',
                 '????????????????????????/????????????',]:
        return text.replace('??????','').replace('??????','').replace('???','').replace(' ','') 
    

    '''
        value = value.replace('##l:','????????????').replace('tel:','????????????').replace('Tel:','????????????').replace('TEL:','????????????')
        value = value.replace('fax:','????????????').replace('Fax:','????????????').replace('FAX:','????????????')
    
    print(text)
    input("")
    
    # ?????? ??? ??????
    value = fuller(value)
    '''
    value_ret = ''
    for c in value:
        if c in text:
            c = c
        elif chr(ord(c)+65248) in text:
            c = chr(ord(c)+65248)
        elif ord('a')<=ord(c) and ord(c)<=ord('z'): #???????????????
            if chr(ord(c)-32) in text: #??????????????? ??????
                c = chr(ord(c)-32)
            elif chr(ord(c)+65248-32) in text: #??????????????? + ?????????
                c = chr(ord(c)+65248-32)
        elif ord('A')<=ord(c) and ord(c)<=ord('Z'): #???????????????
            if chr(ord(c)+32) in text: #??????????????? ??????
                c = chr(ord(c)+65248+32)            
            elif chr(ord(c)+65248+32) in text: #??????????????? + ?????????
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
        data = {'doc':[],'index':[],'ID':[],'Tag':[],'Value':[]}
        for iii,(doc, index, ids, _, masks, sample) in enumerate(dataloader):
            
            output = model(ids)[0]
            prob = F.sigmoid(output)
            
            doc, index, ids, masks, sample = doc[0], index[0], ids[0], masks[0], sample[0]
            
            data['doc'].append(doc)
            data['index'].append(index)
            data['ID'].append(f"{doc}-{index}")
            data['Tag'].append("")
            data['Value'].append("")
            

            for i,tag in enumerate(tags):
                values = []
                for j in range(prob.size(0)):
                    if prob[j,i] > 0.5:
                        values.append(ids[j])

                if len(values)>0:                    
                    value_str = tokenizer.decode(values, skip_special_tokens=True).replace(" ","")
                    if args.postprocess:
                        value_str = post_process(value_str, tag, sample['text'])
                    else:
                        value_str = value_str 

                    # add a tag&value to <Tag> <Value>
                    if data['Tag'][-1] == "":
                        data['Tag'][-1] = "{}".format(tag)
                        data['Value'][-1] = "{}".format(value_str)
                    else:
                        data['Tag'][-1] += ";{}".format(tag)
                        data['Value'][-1] += ";{}".format(value_str)

            #total_dataframe = total_dataframe.append(sample) if isinstance(total_dataframe, pd.DataFrame) else sample

            print(f'\t[Info] [{iii+1}/{len(dataloader)}]', end='   \r')
            
        total_dataframe = pd.DataFrame(data)
            
        print('\t[Info] finish inference ')
            
        # sort dataframe & save results
        total_dataframe = total_dataframe.sort_values(by=['doc','index'], ascending=[True,True])
        #total_dataframe = total_dataframe.drop('Page No', axis=1).drop('Parent Index', axis=1).drop('Is Title', axis=1).drop(
        #                'Is Table', axis=1).drop('id', axis=1).drop('Index', axis=1)
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

    dataset = Cinnamon_Dataset_Testing_v2(f'/media/D/ADL2020-SPRING/project/cinnamon/{args.dev_or_test}/', tokenizer, tags)
    dataloader = DataLoader(dataset,
                             batch_size=1,
                             collate_fn=dataset.collate_fn,
                             shuffle=False)

    inference(args, tokenizer, dataloader)
    
