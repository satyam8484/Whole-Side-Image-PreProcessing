from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import filedialog
#from Main_Logic import *
from tkinter import messagebox





class Ui():
    def __init__(self):
        self.window = Tk()
        self.photo=PhotoImage(file='Fig_1.png')
        self.Button_Frame=Frame(self.window,bd=0,relief=SUNKEN)
        self.fol_path=''
        self.pgbar=''
        self.terminal=''

    def Browse_fol(self):
        fol_path = filedialog.askdirectory()
        print(fol_path)
        self.pgbar.config(value=30)
        #self.Button_Frame(config="watch")
        out_path_split_image=split_image(fol_path,self.pgbar,self.terminal)
        cell_Feature(out_path_split_image,self.pgbar,self.terminal)
        self.pgbar.config(value=100)
        messagebox.showinfo('Done',' Task Completed ')
        #self.Button_Frame(config="arrow")



    def popup(self):
        win = Toplevel()
        win.geometry("500x100+300+300")

        bt = Button(win,text='ok', command=win.destroy)
        bt.pack()


    def set_window(self):
        self.window.title('Whole Side Image Processing')
        self.RWidth = self.window.winfo_screenwidth()
        self.RHeight = self.window.winfo_screenheight()
        self.window.geometry(("%dx%d") % (self.RWidth, self.RHeight))
        self.window.configure()



    def set_header(self):
        f1=Frame(self.window,height=50,bg='black',bd=2,relief=SUNKEN)
        f1.pack(side=TOP,fill=BOTH)

        l1=Label(f1,text='Whole Side Image PreProcessing',bg='black',fg='white',font=("Arial",30,'italic'))
        l1.pack(side=LEFT,fill=BOTH,pady=25)

############   MIDDLE WIDGETS , TERMINAL , IMAGE FRAMES ##########

    def set_image_frame(self):
        image_frame=Frame(self.window)
        image_frame.pack(side=RIGHT)

        img_l1=Label(image_frame,image=self.photo)
        img_l1.pack()


    def set_terminal_frame(self):
        mid_frame=Frame(self.window)
        mid_frame.pack(side=BOTTOM)

        scrollbar = Scrollbar(mid_frame,cursor='hand2',bg='lightblue')
        scrollbar.pack(side=RIGHT, fill=Y)

        self.terminal=Text(mid_frame,width=self.RWidth,height=15,yscrollcommand=scrollbar.set)
        self.terminal.pack(side=BOTTOM,fill=BOTH)
        scrollbar.config(command=self.terminal.yview)

    def set_Widget_frame(self):
        self.Button_Frame.pack(side=BOTTOM, anchor='sw', pady=30)

        self.pgbar = Progressbar(self.Button_Frame, length=200, orient=HORIZONTAL,value=5, mode='determinate')
        self.pgbar.pack(side=BOTTOM,pady=15)

        browse_button = Button(self.Button_Frame, text='Browse', width=10, command=self.Browse_fol,cursor='hand1')
        browse_button.pack(side=BOTTOM)

        label_Button = Label(self.Button_Frame,text='\t Browse the Path of Input Folder Which contain all .svs Files',font=("Courier New",10,'bold') ,fg='black')
        label_Button.pack(side=LEFT,anchor='se',padx=20,pady=5)

        #label_Button_1 = Button(self.Button_Frame,text='\nClick here ...', fg='blue',borderwidth=0,cursor='hand2',font=("Arial",10,'underline'),command=self.popup)
        #label_Button_1.pack(side=LEFT)


############   MIDDLE WIDGETS , TERMINAL , IMAGE FRAMES ##########


    def set_footer(self):
        f2=Frame(self.window,height=50,bg='black',bd=2,relief=SUNKEN)
        f2.pack(side=BOTTOM,fill=BOTH)

        l2=Label(f2,text='@ Copyright- Designed and Develeoped by Arraygen Technology Pvt. Ltd.',bg='black',fg='white')
        l2.pack(fill=BOTH,pady=15)





    def MainLoop_ui(self):
        self.window.mainloop()
