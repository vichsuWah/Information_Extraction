
def get_tags(cinnamon_path):
    tags = set()
    files = glob.glob(f'{cinnamon_path}/ca_data/*')
    for file in files:
        dataframe = pd.read_excel(file, encoding="utf8")
        label_str = filter(lambda i:(type(i) is str), dataframe['Tag'])
        def split(strings):
            out = list()
            for string in strings: 
                out += string.split(";")
            out = [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag)) for tag in out]
            return out
        items = split(label_str)
        tags.update(items)
    return tuple(sorted(list(tags)))

def clean_str(s):
    return str(s).replace('イ．','').replace('ア．','').replace('．','').replace(' ','')

def sub_idx_finder(list1, list2, t=None):            
            if t=='入札件名':
                for i in range(len(list1)-len(list2)+1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if miss < len(list2)/4:
                        find = True
                    if find:
                        return i
            elif t=='需要場所(住所)': #反過來找
                for i in range(len(list1)-len(list2), -1, -1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if miss < len(list2)/6:
                        find = True
                    if find:
                        return i          
            elif t=='質問箇所所属/担当者': #反過來找
                for i in range(len(list1)-len(list2), -1, -1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if miss < len(list2)/4:
                        find = True
                    if find:
                        return i           
            elif t=='質問箇所TEL/FAX': #反過來找
                for i in range(len(list1)-len(list2), -1, -1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if miss < len(list2)/3:
                        find = True
                    if find:
                        return i    
            else: #正向找
                for i in range(len(list1)-len(list2)+1):
                    find = True
                    hit, miss = 0, 0
                    for j in range(len(list2)):
                        if list1[i+j] != list2[j]: 
                            find = False
                            miss += 1
                        else:
                            hit += 1
                    if find:
                        return i                
            return None