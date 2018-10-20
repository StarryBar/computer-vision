from PIL import Image

#imageArray = np.array(Image.open('/Users/KunyuYe/Desktop/Frame/frame0.jpg').convert('L'))

for i in range(7,20):
    im = Image.open('/Users/KunyuYe/Desktop/Frame/%d.jpg' % i)
    im.save("/Users/KunyuYe/Desktop/Frame2/%d.jpg"%(i-7),"JPEG")
'''
for i in range(0,40):
    im = Image.open('/Users/KunyuYe/Desktop/Frame/%d.jpg' % i)
    im.thumbnail((256,256))
    im.save("/Users/KunyuYe/Desktop/Scale/%d.jpg"%i,"JPEG")
'''