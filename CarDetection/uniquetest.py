import copy
import re

def takeFirst(elem):
    return elem[0]

def writetotxt(data,filename):
    b=[str(i) for i in data]
    with open(filename,'w') as f:
        for line in b:
            f.write(line+'\n')
    return 

def readfromtxt(filename):
    b=[]
    with open(filename,'r') as f:
        lines=f.readlines()
        for i in lines:
            for j in ',()':
                i=i.replace(j,' ')
            templine=i.split()
            index,x1,y1,x2,y2,xc,yc=[float(j) for j in templine]
            b.append((index,(x1,y1),(x2,y2),(xc,yc)))
    return b


def unique(frameout):
    frameout.sort(key=takeFirst)
    templist1=[]
    templist2=[]
    i=0
    j=0
    while i<len(frameout)-2:
        tempindex=frameout[i][0]
        for j in range(i+1,len(frameout)):
            if frameout[j][0]!=tempindex:
                if j-i<2:
                    break
                else:                    
                    templist1=copy.deepcopy(frameout[i:j])
                    for k in range(i,j):
                        frameout.pop(i)
                    templist2=centerunique(templist1)
                    for k in templist2:
                        frameout.insert(i,k)
                    i=i+len(templist2)-1
                    break
        i=i+1
    return frameout

def distance(p1,p2):
    (x1,y1)=p1
    (x2,y2)=p2
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def centerunique(frameout):
    templist=[]
    (index_g,(x1_g,y1_g),(x2_g,y2_g),(xg,yg))=frameout[0]
    for i,(index,(x1,y1),(x2,y2),(xc,yc)) in enumerate(frameout):
        if distance((xc,yc),(xg,yg))>20:
            templist.append(frameout[i])
        else:
            x1_g=(x1+x1_g*i)/(i+1)
            x2_g=(x2+x2_g*i)/(i+1)
            y1_g=(y1+y1_g*i)/(i+1)
            y2_g=(y2+y2_g*i)/(i+1)
            xg=(xc+xg*i)/(i+1)
            yg=(yc+yg*i)/(i+1)

    templist.append((index_g,(x1_g,y1_g),(x2_g,y2_g),(xg,yg)))
    i=0
    while i<len(templist)-2:
        (tempxc1,tempyc1)=templist[i][3]
        for j in range(i+1,len(templist)):
            (tempxc2,tempyc2)=templist[j][3]
            if distance((tempxc1,tempyc1),(tempxc2,tempyc2))<20:
                templist.insert(0,templist[i])
                templist.insert(0,templist[j])
                return centerunique(templist)
        i=i+1
    return templist

b=readfromtxt('original.txt')
templist=unique(b)
writetotxt(templist,'uniquetest.txt')
