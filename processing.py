f=open('proteinGroups_Sup.txt','r')
dic={}
f.readline()
lid={}
for l in f:
    sp=l.rstrip().split('\t')
    idl=sp[0].split(';')
    idg=sp[1].split(';')
    inte=sp[2:12]
    for ii,id in enumerate(idg):
        dic[idl[ii]+'\t'+idg[ii]]=inte
        lid[idl[ii]+'\t'+idg[ii]]=''
        
f=open('proteinGroups_pel.txt','r')
dic1={}
f.readline()
for l in f:
    sp=l.rstrip().split('\t')
    idl=sp[0].split(';')
    idg=sp[1].split(';')
    inte=sp[2:12]
    for ii,id in enumerate(idg):
        dic1[idl[ii]+'\t'+idg[ii]]=inte
        lid[idl[ii]+'\t'+idg[ii]]=''

w=open('data.txt','w')
for k in lid:
    if k not in dic:continue
    if k not in dic1:continue
    if dic[k][0]=='0' or dic1[k][4]=='0':
        continue
    dic[k]=[str(float(x)/float(dic[k][0])) for x in dic[k]]
    dic1[k]=[str(float(x)/float(dic1[k][4])) for x in dic1[k]]
    w.write(k+'\t'+'\t'.join(dic[k])+'\t'+'\t'.join(dic1[k])+'\n')

