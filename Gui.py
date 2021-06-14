from tkinter import *
from face_detect import detect
from face_recognition import recognize


windows = Tk(screenName=None,baseName=None,className="First Program")
windows.title("Face Recognition")
windows.geometry('700x700')
name = Entry(windows)
name.pack()
    
def det():
    text = detect(name.get())
    label = Label(windows,text=text)
    label.pack()

def re():
    recognize()
button = Button(windows,text='detect',command=det)
button.pack()
button1 = Button(windows,text='reognize',command=re)
button1.pack()

windows.mainloop()
