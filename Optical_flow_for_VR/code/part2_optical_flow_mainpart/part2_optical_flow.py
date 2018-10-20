
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#note: please run the last two
#--------------------------------------标准代码----输入------------------------
img0w=296
img0h=296

in_path = "./0_1.4/"    #input
out_path="./0_1.4/control_flow.txt"   #event output
rou_path="./0_1.4/frame_analysis.txt"   #frame rou theta output
of_path="./0_1.4/result/"    #file output
file=[]
for i in range(0,17):      #range(60)取前60张图片
    file.append(str(i)+".jpg")             #a serious of pictures  ----file  (1.jpg, 2.jpg, 3.jpg)

print("input finish")



#------------------------------------------------------------------------------------------------
test=np.zeros([20,300,300])
size=30
for i in range(5):
    test[i,10:10+size,150:150+size]=200
for i in range(5,10):
    test[i,10+int((i-4)*15):10+int((i-4)*15)+size,150:150+size]=200
    
for i in range(10,15):
    test[i,10+75:10+75+size,150-int((i-9)*15):150-int((i-9)*15)+size]=200

for i in range(15,20):
    test[i,10+75:10+75+size,150-75:150-75+size]=200
for i in range(test.shape[0]):#IX.shape[0]):
    plt.imsave(in_path+"%d.jpg"%i,test[i])    
   
'''for i in range(10,13):
    test[i,128+10+(i-9)*5:128+10+(i-9)*5+size,50+10+20:50+10+20+size]=200
    
for i in range(17,19):
    test[i,128+10+15-(i-16)*5:128+10+15-(i-16)*5+size,50+10+20-(i-16)*10:50+10+20-(i-16)*10+size]=200'''

#--------------function-------------------------------------------------------
#-------------------------------------边缘提取，IX,IY,IT图，
def edge_laplacian(squ,threthod=0):          #reduce 2+2  (gauss+2,derivative+2)
    squ=np.array(squ)
    C = conv_2d(sm_gauss,squ,padding="zero")          #we cannot use gaussian, because it cause the direction change
    #rgb_lvlC=rgb_kernel(C)
    #showpic_2(rgb_kernel(squ),rgb_lvlC)
    
    edgeX=conv_2d(dx,C,padding="zero")               #reduce two dimension in edge effects x
    edgeY=conv_2d(dy,C,padding="zero")               #reduce two dimension in edge effects y
    edgeXY=np.sqrt(edgeX**2+edgeY**2)    #维度不一样
    angleXY=np.arctan2(edgeY,edgeX)

    #edgeX=calibration(edgeX)
    #edgeY=calibration(edgeY)
    #edgeXY=calibration(edgeXY)   #0~255
    #angleXY=calibration(angleXY) #0~255
    
    edgeXY[edgeXY<threthod]=0          #threthoding=70    
    return edgeX,edgeY,edgeXY,angleXY


#input matrix[t][x][y]
def IXIYIT_group(matrix):              #create 3:-3  edge side-effects
    matrix=np.array(matrix)
    IX=[]
    IY=[]
    IT=[]
    for i in range(matrix.shape[0]-1):
        if i%2==0:
            print("processing rate%.2d%%"%(i/matrix.shape[0]*100))
        tempX,tempY,_,_=edge_laplacian(matrix[i])
        IX.append(list(tempX))
        IY.append(list(tempY))
        IT.append(list(matrix[i+1,:,:]-matrix[i,:,:]))
    print("processing rate100%")
    return np.array(IX),np.array(IY),np.array(IT)

def velocity_kernel(IX,IY,IT,mode="D4"):             # till now create x/y: 4:-4 side-effects   | t: :-1 side-effects
    U=np.zeros(IX.shape)
    V=np.zeros(IY.shape)
    for t in range(IT.shape[0]):
        if t%2==0:
            print("processing rate%.2d%%"%(t/IT.shape[0]*100))
        for x in range(1,IX.shape[1]-1):
            for y in range(1,IY.shape[2]-1):
                if mode=="D4":
                    W=[[IX[t,x,y],IY[t,x,y]],
                      [IX[t,x-1,y],IY[t,x-1,y]],
                      [IX[t,x+1,y],IY[t,x+1,y]],
                      [IX[t,x,y-1],IY[t,x,y-1]],
                      [IX[t,x,y+1],IY[t,x,y+1]]]
                    Y=[-IT[t,x,y],
                       -IT[t,x-1,y],
                       -IT[t,x+1,y],
                       -IT[t,x,y-1],
                       -IT[t,x,y+1]]
                if mode=="D8":
                    W=[[IX[t,x,y],IY[t,x,y]],
                      [IX[t,x-1,y],IY[t,x-1,y]],
                      [IX[t,x+1,y],IY[t,x+1,y]],
                      [IX[t,x,y-1],IY[t,x,y-1]],
                      [IX[t,x,y+1],IY[t,x,y+1]],
                      [IX[t,x-1,y-1],IY[t,x-1,y-1]],
                      [IX[t,x-1,y+1],IY[t,x-1,y+1]],
                      [IX[t,x+1,y-1],IY[t,x+1,y-1]],
                      [IX[t,x+1,y+1],IY[t,x+1,y+1]]]
                    Y=[-IT[t,x,y],
                       -IT[t,x-1,y],
                       -IT[t,x+1,y],
                       -IT[t,x,y-1],
                       -IT[t,x,y+1],
                       -IT[t,x-1,y-1],
                       -IT[t,x-1,y+1],
                       -IT[t,x+1,y-1],
                       -IT[t,x+1,y+1]]
                aff=np.linalg.lstsq(W,Y,rcond=-1)[0]    #rcond...?????waht???????????????
                
                u=aff[0]
                v=aff[1]
                U[t,x,y]=u
                V[t,x,y]=v
    print("processing rate100%")

    return U,V
#------------------------------to transform each image frame into one scalar,pick threthod, and stack the selected points---
def avg_rou_theta(rou,theta,threthod,interval,mode="avg"):        #rou&theta image threthod,    theta interval
    def categorize(array,interval):   #interval as 30,60... ....
        interval=interval/180*np.pi
        for i in range(array.shape[0]):
            array[i]=round(array[i]/interval)*interval
        return array
    def avg_theta(rou,theta,threthod):
        mean_rou=np.zeros(rou.shape[0])
        mean_theta=np.zeros(rou.shape[0]) 
        for i in range(rou.shape[0]):      
            chosen=(rou[i]>threthod[0])&(rou[i]<threthod[1])
            if rou[i,chosen].shape[0]==0:
                mean_rou[i]=0
                mean_theta[i]=0
            else:
                mean_rou[i]=np.mean(rou[i,chosen])
                temp=theta[i,chosen] #selected

                temp_mean_cos=np.mean(np.cos(temp)*rou[i,chosen])
                temp_mean_sin=np.mean(np.sin(temp)*rou[i,chosen])
                mean_theta[i]=np.arctan2([temp_mean_sin],[temp_mean_cos])
        return mean_rou,mean_theta

    def pool_theta(rou,theta,threthod):
        mean_rou=np.zeros(rou.shape[0])
        mean_theta=np.zeros(rou.shape[0]) 
        for i in range(rou.shape[0]):
            if i%2==0:
                print("pooling process:%d%%"%(i/rou.shape[0]*100))
            chosen=(rou[i]>threthod[0])&(rou[i]<threthod[1])
            if rou[i,chosen].shape[0]==0:
                mean_rou[i]=0
                mean_theta[i]=0
            else:
                mean_rou[i]=np.mean(rou[i,chosen])
                angle_dict=np.zeros(360)
                angle=(np.arctan2(np.sin(theta[i,chosen]),np.cos(theta[i,chosen]))/np.pi*180).astype(np.int)
                for j in range(angle.shape[0]):
                    if angle[j]>=0:
                        angle_dict[angle[j]]+=(rou[i,chosen])[j]
                    else:
                        angle_dict[angle[j]+360]+=(rou[i,chosen])[j]
                if np.argmax(angle_dict) < 180:
                    mean_theta[i]=np.argmax(angle_dict) 
                else:
                    mean_theta[i]=np.argmax(angle_dict)-360
        print("pooling process:100%")
        return mean_rou,mean_theta/180*np.pi
    #------------------main part------------
    if mode=="avg":
        mean_rou,mean_theta=avg_theta(rou,theta,threthod)
    if mode=="pool":
        mean_rou,mean_theta=pool_theta(rou,theta,threthod)
    mean_theta=categorize(mean_theta,interval)
    res=np.zeros([2,mean_rou.shape[0]])
    res[0]=mean_rou
    res[1]=mean_theta/np.pi*180
    return res




#arr is a two dimentional numpy.array, the first dimention is velocity and second dimention is angle
def caseCalculation(arr,threshold=[0,0]):

    cols = arr.shape[1]
    start = -1
    for i in range(cols):
        if start < 0:
            if arr[0][i] > threshold[0]:
                start = i
        else:
            end=i
            if arr[0][i] <= threshold[1]:
                end = i-1
                break
    #start = 0
    #end = cols-1
    ave_velocity = np.sum(arr[0][start:(end+1)])/(end-start+1)
    print("ave",arr[0][start:(end+1)],start,end)
    directionChangeTimes = []
    angleBeforeChange=[]
    single_angle=0

    for i in range(start+1,end+1):
        if arr[1][i] != arr[1][i-1] :
            directionChangeTimes.append(arr[1][i]-arr[1][i-1])
            single_angle=arr[1][i-1]
            angleBeforeChange.append(single_angle)
    if len(directionChangeTimes)==0:
        single_angle=arr[1][start]
    times=len(directionChangeTimes) #times=0,1
    
    if times==0:                #angle
        mode=single_angle//45+1    #-45,0,45,90->0,1,2,3
        event=mode
    if times==1:
        for i in range(times):
            if directionChangeTimes[i]<0:                                      #before -- #(shift)
                mode=angleBeforeChange[i]//90*4+directionChangeTimes[i]//45+2   #0--(-90,-45,45,90)->4,5,6,7
            if directionChangeTimes[i]>0:
                mode=angleBeforeChange[i]//90*4+directionChangeTimes[i]//45+1   #90--(-90,-45,45,90)->8,9,10,11
        event=4+mode
    #event=0,1,2,3,4,5,6,7,8,9,10,11
    
    return ave_velocity,event #return a tuple, the first element is average velocity, second ele is directionChangeTimes 


def write_file(matrix,name='control_flow.txt'):
    file = open(name, 'w')
    for i in range(matrix.shape[0]):
        for index,element in enumerate(matrix[i]):
            event=str(element)
            file.writelines(event)
            if index !=matrix.shape[1]-1:
                file.writelines(",")
        file.writelines('\n')
    file.close()   




#----------------donnot modify this-------------modify everything above
#---------------------read the image--------input path="e://python3/cv/"   file="raw.jpg" height,width of image
#-------------------------------------------output type=np.array(), shape= img[height][width]
def preprocess(path,file,imgh,imgw):   #path="e://.." file="...png"
    def grey_kernel(matrix,imgh=0,imgw=0):   #greying the 1D RGB image   
        elements=matrix.shape[0]*matrix.shape[1]
        if elements>=imgh*imgw*3:   #a three channel image
            res=np.zeros(matrix.shape[0],dtype=np.uint8)
            for i in range(matrix.shape[0]):
                grey=np.uint8(round(0.299*matrix[i,0]+0.587*matrix[i,1]+0.114*matrix[i,2]))#matrix[i,0]#
                res[i]=grey
        elif elements==imgh*imgw:    #a grey image
            res=matrix
        return res

    def squ_image(matrix,height,weight):
        res=np.array(matrix).reshape(height,weight)
        return res
    
    img = Image.open(path+str(file))
    imageMatrix=np.matrix(img.getdata())
    grey_array=grey_kernel(imageMatrix,imgh,imgw)
    grey_square=squ_image(grey_array,imgh,imgw)
    #showpic_1(rgb_kernel(grey_square))
    return grey_square    #return size=img[x][y]   typ=np.array()
    

#-------------------show the image---------input type=np.array()  shape=img[height][width]
#------------------------------------------output type=np.array()   shape=img[height][width][channel]
def deprocess(img,enhence="yes",threthod=[-3000,3000]):    #img_type=img[x][y]   
    def rgb_kernel(squ_array):   #RGBing the 2D grey image
        res=np.zeros([squ_array.shape[0],squ_array.shape[1],3],dtype=np.uint8)                        #NO NEED TOcalibration for display.                                                                          #squ_array-squ_array.min() :× squ_array can never be negative  
        for i in range(squ_array.shape[0]):
            for j in range(squ_array.shape[1]):
                res[i,j,0]=np.uint8(round(squ_array[i,j]))
                res[i,j,1]=np.uint8(round(squ_array[i,j]))
                res[i,j,2]=np.uint8(round(squ_array[i,j])) 
        return res
    
    if enhence=="yes":
        img=calibration(img,threthod)
    
    res=rgb_kernel(img)
    
    #showpic_1(res)
    return res

def calibration(squ_array,threthod):   #RGBing the 2D grey image
    #scale=squ_array.max()-squ_array.min()                          #calibration for display.
    squ_array[squ_array<threthod[0]]=threthod[0]
    squ_array[squ_array>threthod[1]]=threthod[1]
    scale=threthod[1]-threthod[0]
    '''if(scale==0):
        new_array=squ_array+128
        print("fail")
    else:'''
    new_array=squ_array/scale*256+128
    return new_array    #type=np.array

#---------------------------picture display------------------------
def showpic_1(target0,name0="fg1"):
    fig=plt.figure(figsize=(15,5))
    plt.title(name0)
    plt.imshow(target0,cmap=plt.cm.gray)
def showpic_2(target0,target1,name0="fg1",name1="fg2"):       #show picture and save picture                                       
    fig=plt.figure(figsize=(15,5))    
    fig.add_subplot(121)
    plt.title(name0)
    plt.imshow(target0,cmap=plt.cm.gray)
    fig.add_subplot(122)
    plt.title(name1)
    plt.imshow(target1,cmap=plt.cm.gray)   
def showpic_3(target0,target1,target2,name0="fg1",name1="fg2",name2="fg3"):       #show picture and save picture                                       
    fig=plt.figure(figsize=(15,5))    
    fig.add_subplot(131)
    plt.title(name0)
    plt.imshow(target0)
    fig.add_subplot(132)
    plt.title(name1)
    plt.imshow(target1,cmap=plt.cm.gray) 
    fig.add_subplot(133)
    plt.title(name2)
    plt.imshow(target2,cmap=plt.cm.gray) 
def showpic_4(target0,target1,target2,target3,name0="fg1",name1="fg2",name2="fg3",name3="fg4"):       #show picture and save picture                                       
    fig=plt.figure(figsize=(15,5))    
    fig.add_subplot(141)
    plt.title(name0)
    plt.imshow(target1,cmap=plt.cm.gray)
    fig.add_subplot(142)
    plt.title(name1)
    plt.imshow(target2,cmap=plt.cm.gray)
    fig.add_subplot(143)
    plt.title(name2)
    plt.imshow(target3,cmap=plt.cm.gray)
    fig.add_subplot(144)
    plt.title(name3)
    plt.imshow(target4,cmap=plt.cm.gray)    

#---------------------------conv2d------------------------------------------------------
def conv_2d(A,B,mode=1,padding="no"):   #A=kernel, B=img ,    padding makes same size
    if(padding=="zero"):
        
        C=np.zeros(B.shape)
        
        tempC=np.zeros([B.shape[0]+A.shape[0] - 1 , B.shape[1]+A.shape[1] - 1])
        xshift=int((A.shape[0] - 1 )/2)
        yshift=int((A.shape[1] - 1 )/2)
        if xshift!=0 and yshift!=0:
            tempC[xshift:-xshift,yshift:-yshift]=B
        if xshift==0 and yshift!=0:
            tempC[:,yshift:-yshift]=B
        if yshift==0 and xshift!=0:
            tempC[xshift:-xshift,:]=B
        B=tempC
        
    if(padding=="reflect"):
        tempC=np.zeros([B.shape[0]+A.shape[0] - 1 , B.shape[1]+A.shape[1] - 1])
        xshift=int((A.shape[0] - 1 )/2)
        yshift=int((A.shape[1] - 1 )/2)
        C=np.zeros(B.shape)
        
        tempC[xshift:-xshift,yshift:-yshift]=B
        tempC[xshift:-xshift,-yshift:]=B[:,:yshift]
        tempC[xshift:-xshift,:yshift]=B[:,-yshift:]
        tempC[:xshift,:]=tempC[-2*xshift:-xshift,:]
        tempC[-xshift:,:]=tempC[xshift:2*xshift,:]    
        
        B=tempC
    if(padding=="no"):
        C=np.zeros([B.shape[0]-A.shape[0] + 1 , B.shape[1]-A.shape[1] + 1],dtype=float)
    if(mode==1):
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C[i,j]=(A * B[i:i+A.shape[0],j:j+A.shape[1]]).sum()    #np.dot(A,Bxxx.T).trace()
    
    if(mode==2): 
        A=A-A.mean()                                                 #normalize the template ADD-square-SQRT,to match correlation<=1
        A=A/np.sqrt((A**2).sum())
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                TEMP=B[i:i+A.shape[0],j:j+A.shape[1]]                #normalize the image window
                TEMP=TEMP-TEMP.mean()
                TEMP=TEMP/np.sqrt((TEMP**2).sum())
                C[i,j]=(A * TEMP).sum()

    return C


#----------------------------------------smooth kernel--------------------------------------------
sm_filter33=np.array(np.ones([3,3],dtype=float)) #python:    Matrix *Array *:stands for dot;  array*array:*stands for .*!!!!
sm_filter33/=9

sm_filter55=np.array(np.ones([5,5],dtype=float))
sm_filter55/=25

def smooth_filter(size):
    sm_filter=np.array(np.ones([size,size]),dtype=float)
    sm_filter/=(size**2)
    return sm_filter

#-------------------------------------gaussian kernel-------------------------------------------------------
sm_gauss=np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=float)
sm_gauss/=16
def gaussian_filter(size):
    gau=np.zeros([size,size],dtype=float)
    thegma=np.sqrt(2.245/np.pi)
    for x in range(size):
        for y in range(size):
            gau[x,y]=1/(2*np.pi*thegma**2)*np.exp(-((x-int(size/2))**2+(y-int(size/2))**2)/(2*thegma**2))
    return gau/gau.sum()

#-----------------------------------second derivative-------------------------------------------------
dx=np.array([[-0.5],[0],[0.5]])  #in order to match matrix axis
dy=np.array([[-0.5,0,0.5]])      #in order to match matrix axis

#-----------------------------------laplacian kernel--------------------------------------------------
peak_lp=np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]],dtype=float)
peak_lp/=8



# In[7]:


#--------------------------main function-----------------------------------------------------------
#----------------------------create three cube for IX,IY,IT-----------处理---------------
                           #squ1代表一帧图片
squ=np.zeros([len(file),296,296]) #squ=np.zeros([len(file),img0h,img0w])     #store
for i in range(len(file)):
    if i%2==0:
        print("read file:%d%%"%(i/len(file)*100))
    squ[i]=preprocess(in_path,file[i],img0h,img0w)   #squ1  type=np.array,  size=img[img1h][img1w]
print("read file:100%")
             

IX,IY,IT=IXIYIT_group(squ)                #IX type=np.array,      size=img[time][img1h][img1w]
print("IXIYIT finish")
U,V=velocity_kernel(IX,IY,IT,mode="D4")            # 4:-4 side-effects    0:-1 side-effects in time
print("UV finish")
rou=np.sqrt(U**2+V**2)
theta=np.arctan2(V,U)
print("ROU,THETA finish")

      

'''IX,IY,IT=IXIYIT_group(test)                #IX type=np.array,      size=img[time][img1h][img1w]
print("IXIYIT finish")
U,V=velocity_kernel(IX,IY,IT)            # 4:-4 side-effects    0:-1 side-effects in time
print("UV finish")
rou=np.sqrt(U**2+V**2)
theta=np.arctan2(V,U)
print("ROU,THETA finish")
res=avg_rou_theta(rou,theta,[10,300],30)
'''


# In[8]:


'''print(np.uint8(round(32.000000000000014)))
for i in range(19):
    print("U%d"%i,deprocess(U[i],threthod=[-128,128]).max(),deprocess(U[i],threthod=[-128,128]).min())
    print("V%d"%i,deprocess(V[i],threthod=[-128,128]).max(),deprocess(V[i],threthod=[-128,128]).min())
    print("rou%d"%i,rou[i].max(),rou[i].min())

'''


# In[9]:


print(rou[5].max(),rou[5].min())
print(theta[5].max(),theta[5].min())
resaa=avg_rou_theta(rou,theta,threthod=[1,3000],interval=15,mode="avg")
print(resaa)
event=caseCalculation(resaa)
print("event finish")
event=np.array(event,dtype=int).reshape([1,2])
print(event)
#--------------------------output-------------------------------------

write_file(event,name=out_path)

'''U[U>3000]=3000
U[U<-3000]=-3000
V[V>3000]=3000
V[V<-3000]=-3000
print(U[1].max(),U.min())
'''
for i in range(IX.shape[0]):#IX.shape[0]):
    plt.imsave(str(of_path)+"U%d.jpg"%i,deprocess(U[i],enhence="yes",threthod=[-128,128]))
    plt.imsave(str(of_path)+"V%d.jpg"%i,deprocess(V[i],enhence="yes",threthod=[-128,128]))
res=resaa.transpose()
write_file(res.astype(int),name=rou_path)
print("write finish")


# In[ ]:


'''#--------------------------display-------------------显示------------------
for i in range(IX.shape[0]):#IX.shape[0]):
    showpic_3(deprocess(IX[i]),deprocess(IY[i]),deprocess(IT[i]),"IX%d"%i,"IY%d"%i,"IT%d"%i)
    showpic_2(deprocess(U[i]),deprocess(V[i]),"U%d"%i,"V%d"%i)    
    plt.imsave('./result/U%d.jpg'%i,deprocess(U[i],enhence="yes"))
    plt.imsave('./result/V%d.jpg'%i,deprocess(V[i],enhence="yes"))
for i in range(rou.shape[0]):#rou.shape[0]):
    showpic_2(deprocess(rou[i]),deprocess(theta[i]),"rou%d"%i,"theta%d"%i)
'''


# In[ ]:


#-------------------------------------debugg!!!!!!!----------------------
#-------------------------------特殊修改velocity kernel-----------------------
'''def velocity_kernel(IX,IY,IT):             # till now create x/y: 4:-4 side-effects   | t: :-1 side-effects
    U=np.zeros(IX.shape)
    V=np.zeros(IY.shape)
    for t in range(IT.shape[0]):
        for x in range(2,IX.shape[1]-2,5):
            for y in range(2,IY.shape[2]-2,5):
                
                W=[[IX[t,x,y],IY[t,x,y]],
                  [IX[t,x-1,y],IY[t,x-1,y]],
                  [IX[t,x+1,y],IY[t,x+1,y]],
                  [IX[t,x,y-1],IY[t,x,y-1]],
                  [IX[t,x,y+1],IY[t,x,y+1]]]
                Y=[-IT[t,x,y],-IT[t,x-1,y],-IT[t,x+1,y],-IT[t,x,y-1],-IT[t,x,y+1]]
                aff=np.linalg.lstsq(W,Y)[0]
                u=aff[0]
                v=aff[1]
                U[t,x-2:x+3,y-2:y+3]=np.ones([5,5])*u
                V[t,x-2:x+3,y-2:y+3]=np.ones([5,5])*v
    return U,V'''
'''test1=np.zeros([13,13])
test1[5:7,5:7]=1000
test1=list(test1)
test2=np.zeros([13,13])
test3=np.zeros([13,13])
shiftx=3
shifty=0
shiftxx=3
shiftyy=5
test2[5+shiftx:7+shiftx,5+shifty:7+shifty]=1000
test3[5+shiftxx:7+shiftxx,5+shiftyy:7+shiftyy]=1000
#test2[5:8,5:8]=test1[4:7,4:7]
test2=list(test2)
mm=[]

mm.append(test1)
mm.append(test2)
mm.append(test3)
IX1,IY1,IT1=IXIYIT_group(mm)                #IX type=np.array,      size=img[time][img1h][img1w]
U1,V1=velocity_kernel(IX1,IY1,IT1)            # 4:-4 side-effects    0:-1 side-effects in time
rou1=np.sqrt(U1**2+V1**2)
theta1=np.arctan2(V1,U1)
res1=avg_rou_theta(np.array(rou1),np.array(theta1),[0,3000],1)
average1,times1=caseCalculation(res1)
#res1=res1.transpose().astype(int)
print(res1)
print(average1,times1)
#write_file(np.array([average1,times1]),name="./skew/control.txt")
print(np.array(mm))
print("IX",IX1.astype(np.int))
print("IY",IY1.astype(np.int))
print("IT",IT1.astype(np.int))
print("U",U1.astype(int))
print("V",V1.astype(int),np.mean(V1[0]))
print("r",rou1.astype(np.int))
print("t",(theta1/np.pi*180).astype(np.int))
print("mean rou and mean theta")
'''
#----------------------------debug2-----------------
'''temp0=1
temp1=21

showpic_2(deprocess(rou[temp0]),deprocess(rou[temp1]),"rou","theta")
mean_rou,mean_theta=avg_rou_theta(np.array([rou[temp0],rou[temp1]]),np.array([theta[temp0],theta[temp1]]),[10,300],30)
print(mean_rou,mean_theta/np.pi*180)'''
'''aff=np.linalg.lstsq([[500.0, 0.0], [0.0, 0.0], [500.0, -500.0], [500.0, 0.0], [0.0, 0.0]],[-0.0, -0.0, 1000.0, -0.0, -0.0],rcond=-1)[0]

aff=np.linalg.lstsq([[500.0, 0.0], [0.0, 0.0], [500.0, -500.0], [500.0, 0.0], [0.0, 0.0]],[-1000.0, -0.0, -0.0, -1000.0, -0.0],rcond=-1)[0]
print(aff)
for i in range(3):
    plt.imsave('./result/U%d.jpg'%i,deprocess(rou1[i]))'''

