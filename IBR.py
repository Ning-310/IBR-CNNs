import numpy as np
import matplotlib.pyplot as plt
import cv2,os

x0=[0,1,2,3,4]
f=open('data.txt','r')
for i,l in enumerate(f):
    if i==0:continue
    l=l.rstrip()
    sp1=l.split('\t')
    sp0=[float(xx) for xx in sp1[2:23]]
    sp=sp1[:2]+sp0
    for s in sp0:
        if s>3:
            print(s,l)
    x=sp[2:7]
    x1=sp[7:12]
    y=sp[12:17]
    y1=sp[17:22]

    for aa in [4]:
        if os.path.exists('graph'+str(aa)+'/'+sp1[34]+' '+sp[1]+' '+sp[0]+".png"):continue
        fig,ax = plt.subplots(figsize=(0.5,0.5))
        #fig.set_facecolor('#000000')
        ax=plt.subplot(axisbg='#000000')
        plt.axis([0,4,-0.1,3.3])
        #plt.show()
        aa=str(aa)
        x0,x,x1=np.array(x0),np.array(x),np.array(x1)
        y,y1=np.array(y),np.array(y1)

        frame = plt.gca()

        plt.plot(x0, x, 'g',alpha=1, linewidth=1)
        plt.plot(x0, x1, 'r', alpha=1,linewidth=1)

        plt.fill_between(x0,(x),x1,where= x > x1,facecolor ='y',alpha=0.5)
        plt.fill_between(x0,(x),x1,where= x1 >x,color='b',alpha=0.5)

        if not os.path.exists('graph'+aa+'/'):
            os.makedirs('graph'+aa+'/')
        plt.savefig('graph'+aa+'/'+sp[1]+' '+sp[0]+".png",dpi=200)
        img = cv2.imread('graph'+aa+'/'+sp[1]+' '+sp[0]+".png")
        print(img.shape)
        cropped = img[13:88, 14:89]
        cv2.imwrite('graph'+aa+'/'+sp[1]+' '+sp[0]+".png", cropped)

