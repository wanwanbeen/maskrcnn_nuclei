__authors__="Jie Yang and Xinyang Feng"

###########################################
# mask2rle after postprocessing
###########################################

import numpy as np
import glob
import pandas as pd

color_id = glob.glob('~/output/*.npy')
for ids in color_id:
    print ids
	
def rle_encoding(x):
    dots = np.where(x.T.flatten()==1)[0]
    run_lengths=[]
    prev=-2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1,0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rle(x,cutoff = 0.5):
    yield rle_encoding(x>cutoff)

HE_id = []
	
rles = []
test_id = []
for ids in color_id:
    idno = ids.split('/')[-1][:-9]
    if idno not in HE_id:
        M=np.load(ids)
        V=np.unique(M)
        V=V[1:]
        mask=np.zeros([M.shape[0],M.shape[1],len(V)])
        for n in range(len(V)):
            tmp = np.zeros([M.shape[0],M.shape[1]])
            tmp[M == V[n]]=1
            mask[:,:,n]=tmp
        print mask.shape, idno
        for n in range(mask.shape[2]):
            rle = list(prob_to_rle(mask[:,:,n]))
            rles.extend(rle)
            test_id.append(idno)
    else:
        rles.extend([(0,0)])
        test_id.append(idno)
	
sub=pd.DataFrame()
sub['ID']=test_id
sub['RLE']=pd.Series(rles).apply(lambda x:' '.join(str(y) for y in x))
sub.to_csv('RLE-test-final.csv',index=False)