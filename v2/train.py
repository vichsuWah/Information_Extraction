try:
    from model import *
    from utils.metrics import metrics
except Exception as inst:
    from v2.model import *
    from v2.utils.metrics import metrics 

import os
import torch
import torch.nn as nn

def _run_train(args, model, criterion, optimizer, dataloader):
    
    model.train()
    
    total_loss, acc, f1 = 0, None, None  
    for idx, (ids, labels, masks) in enumerate(dataloader):
        b,l = ids.shape
        
        optimizer.zero_grad()
       
        predict = model(ids.cuda())
        
        loss = criterion(predict, labels.type_as(predict).cuda())
        #loss = torch.masked_select(loss, masks.expand(b,l,20).cuda())
        loss = loss.mean()
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()*b
        acc, f1 = metrics(predict, labels, acc, f1, idx)
        print("\t[{}/{}] train loss:{:.3f} acc:{:.2f} f1:{:.2f}".format(
                                            idx+1,
                                            len(dataloader),
                                            total_loss/(idx+1)/b,
                                            acc,
                                            f1),
                                    end='   \r')

    return {'loss':total_loss/len(dataloader.dataset), 'acc':acc, 'f1':f1}
    
def _run_eval(args, model, criterion, dataloader):
 
    model.eval()
    
    with torch.no_grad():
        total_loss, acc, f1 = 0, None, None 
        for idx, (ids, labels, masks) in enumerate(dataloader):
            b,l = ids.shape

            predict = model(ids.cuda())

            loss = criterion(predict, labels.type_as(predict).cuda())
            #loss = torch.masked_select(loss, masks.expand(b,l,20).cuda())
            loss = loss.mean()
        
            total_loss += loss.item()*b
            acc, f1 = metrics(predict, labels, acc, f1, idx)
            print("\t[{}/{}] valid loss:{:.3f} acc:{:.2f} f1:{:.2f}".format(
                                            idx+1,
                                            len(dataloader),
                                            total_loss/(idx+1)/b,
                                            acc,
                                            f1),
                                    end='   \r')     

    return {'loss':total_loss/len(dataloader.dataset), 'acc':acc, 'f1':f1}

def train(args, train_dataloader, valid_dataloader):
    torch.manual_seed(987)
    torch.cuda.manual_seed(987)
    
    if args.model == 'naive':
        model = Model()
    elif args.model == 'blstm':
        model = Model_BLSTM() 
    model.load(args.load_model).cuda() 
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40]*20), reduction='none').cuda() 
     
    optimizer = torch.optim.AdamW(list(model.parameters()), 
                                  lr=args.lr,
                                  eps=1e-8 )

    best_f1 = 0
    for epoch in range(1,args.epoch+1):
        print(f' Epoch {epoch}')
            
        train_log = _run_train(args, model, criterion, optimizer, train_dataloader)
        print("\t[Info] Train avg loss:{:.4f} acc:{:.2f} f1:{:.2f}".format(
                            train_log['loss'], train_log['acc'] ,train_log['f1']))
        
        valid_log = _run_eval(args, model, criterion, valid_dataloader)
        print("\t[Info] Valid avg loss:{:.4f} acc:{:.2f} f1:{:.2f}".format(
                            valid_log['loss'], valid_log['acc'], valid_log['f1']))
        
        ## Save ckpt
        if epoch%5==0 or epoch==args.epoch:
            model.save(epoch, train_log, valid_log, args.save_path)
      
        ## Update learning rate
        warmup = 0 
        if valid_log['f1']>=best_f1: # 更新best f1
            best_f1 = valid_log['f1']
        elif warmup<3:
            warmup += 1
        elif args.decline_lr: # 更改lr
            warmup = 0
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2
                if param_group['lr'] < 1e-6:
                    param_group['lr'] = 1e-6 
        
        print('\t--------------------------------------------------------')

