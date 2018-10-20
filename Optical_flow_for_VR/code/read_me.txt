Our code is in four parts

----------------------------------------------------------------
The first part is in VideoProcessCode folder
tkint4er_UI.py is used to generate the video.

videoProcess.py is used to select frames from the video.

We can use scale.py to scale the image(frame).


--------------------------------------------------------------
The second part is to do optical flow:
all the input should be modified by yourself, you need to change the [a,b,c,d,e] part of the code to run
##############################################################
img0w=296[a]
img0h=296[b]

in_path = "./[c]/"    #input
out_path="./[c]/control_flow.txt"   #event output
rou_path="./[c]/frame_analysis.txt"   #frame rou theta output
of_path="./[c]/result/"    #file output
file=[]
for i in range(0,[d]):      #range(60) fetch the first 60 frames from the video
    file.append(str(i)+".jpg")             #a serious of pictures  ----file  (1.jpg, 2.jpg, 3.jpg)

print("input finish")
#############################################################
---------------------------------------------------------------
The third part is for visualization

CreateConstrintsList(VImage, UImage, resulotion, threshold)
function: generate constraint for velocityFiled
output: a constraint list


velocityFiled(SImage, constraintList, resulotion);
function: interpolate and plot the arrow on the SImage

Require: UImage, VImage, SImage must be square
resulotion decided how large each arrows.


--------------------------------------------------------------
The forth part is to use the following table to control the virtual object
   
Input   Output
-------------------------------
motion 	 Function
-45      Flat shading
0        Exponential Square Fog
45       No Effect
90       Add Shadow
vt-90    Point light o90
vt-45    Spot light on
vt45     Multi Light effect
vt90     Wire Frame
hz-90    Fog Off
hz-45    Linear Fog
hz45     Exponential Fog
hz90     Transparent Shadow
---------------------------------
