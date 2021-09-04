import torch
import torch.nn.functional as F

def metrics(_predict, _label, acc=None, f1=None, count=None):
    acc = 0 if acc is None else acc
    f1  = 0 if f1 is None else f1 
    count = 0 if count is None else count 

    b = _predict.size(0)

    ## Accuracy     
    b_acc = accuracy(_predict, _label)
    acc = (acc*count + b_acc*1) / (count + 1)

    ## F1 score    
    b_f1 = f1_score(_predict, _label)
    f1 = (f1*count + b_f1*1) / (count + 1)
    
    return acc, f1

def f1_score(predict, target, mask=None):

    predict, target = predict.detach().cpu(), target.detach().cpu()
    
    if mask is not None:
        mask = mask.detach().cpu()

        predict = torch.masked_select(predict, mask)
        target  = torch.masked_select(target, mask)

    predict = (F.sigmoid(predict))#>0.1).int()
    target  = (target>0.5).int()

    tp = (predict*target).sum().item()
    fp = (predict*(1-target)).sum().item()
    fn = ((1-predict)*target).sum().item()

    recall = tp/(tp+fn+1e-6)
    precis = tp/(tp+fp+1e-6)
    f1 = 2*recall*precis/(recall+precis+1e-6)

    return f1

def accuracy(predict, target, mask=None, position=False):

    predict, target = predict.detach().cpu(), target.detach().cpu()
    
    if position:
        predict = torch.argmax(predict, dim=1)
        target = torch.argmax(target, dim=1)
        
        acc = (predict==target).sum().item() / (predict.nelement()+1e-6)

        return acc 
    '''
    if mask is not None:
        mask = mask.detach().cpu()

        predict = torch.masked_select(predict, mask)
        target  = torch.masked_select(target, mask)
    '''

    predict = F.sigmoid(predict)
    
    acc = ((predict>0.5)==(target>0.5)).sum().item() / (predict.nelement()+1e-6)

    return acc


