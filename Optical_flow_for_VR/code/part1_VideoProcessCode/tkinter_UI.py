import tkinter
import time

window = tkinter.Tk()
window.title('ball movement')
#window.geometry('500x500')
window.geometry('150x150')

#canvas = tkinter.Canvas(window,height=500,width=500)
bgcolor = '#%02x%02x%02x' % (68, 1, 84)
canvas = tkinter.Canvas(window,height=150,width=150,bg=bgcolor)

canvas.pack()
cl = '#%02x%02x%02x' % (253, 231, 36)
#window.mainloop()
for j in range(100):
    #oval = canvas.create_oval(10,430,50,470,fill='red')
    #oval = canvas.create_rectangle(65,10,85,30,fill='red',outline='red')

    oval = canvas.create_rectangle(75,5,90,20,fill=cl,outline=cl)
    for i in range(37):
        canvas.move(oval,0,1)
        window.update()
        time.sleep(0.01)
    
    for j in range(37):
        canvas.move(oval,1,0)
        window.update()
        time.sleep(0.01)
    
    '''
    for i in range(100):
        canvas.move(oval,1,0)
        window.update()
        time.sleep(0.01)
    
    for j in range(100):
        canvas.move(oval,0,1)
        window.update()
        time.sleep(0.01)
    '''
    canvas.delete(oval)


