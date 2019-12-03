#based on: https://github.com/abhishek305/ProgrammingKnowlegde-Tkinter-Series/blob/master/10th/Slider%20and%20color%20choos.py
from tkinter import *
from tkinter import ttk, colorchooser
from collections import namedtuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageTk


class main:
    point = namedtuple('Point', ['x', 'y'])
    point_list = []
    
    #angle detection variables and CONSTANTS
    current_angle = None
    WINDOW_FOR_ANGLE_DETECTION = 15
    ANGLE_MIN_LIMIT = 140
    ANGLE_MAX_LIMIT = 220
    ACCUTE_ANGLE_TRIGGER = 3  # number of deteccted angles in the windows to trigger an "accute angle detected" routine
    number_of_accute_angles = 0
    accute_angle_cleared = False # indicates that the first, ascending line was detected al removed 

    #aspect ratio variables and constant
    margin_left = 0
    margin_rigth = 0
    margin_top = 0
    margin_bottom = 0
    rectangle = None
    RATIO_TRIGGER = 1.4
    minimun_ratio_trigered = False

    #for image processing
    scaled_image = None
    roi_image = None



    def __init__(self,master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.check_live = BooleanVar()
        self.check_live.set(True) 
        self.highlight_or_clear = IntVar(value=2)
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
        
        

    def paint(self,e):
        RECTANGLE_MARGIN = 5

        self.point_list.append(self.point(x = e.x,y = e.y))
        
        if len(self.point_list) > self.WINDOW_FOR_ANGLE_DETECTION:
            mid_point = -1 - (round(self.WINDOW_FOR_ANGLE_DETECTION / 2))
            start_point = -1 - self.WINDOW_FOR_ANGLE_DETECTION
            self.current_angle = self.get_angle(self.point_list[start_point],self.point_list[mid_point],self.point_list[-1])
        
        if self.old_x and self.old_y:
            
            self.c.create_line(self.point_list[-2].x ,self.point_list[-2].y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

            if self.current_angle is not None \
                 and  (self.current_angle >= self.ANGLE_MAX_LIMIT \
                       or self.current_angle <= self.ANGLE_MIN_LIMIT):
                
                self.number_of_accute_angles +=1
                if self.number_of_accute_angles >= self.ACCUTE_ANGLE_TRIGGER \
                    and self.accute_angle_cleared == False:
                    self.clear_path( (len(self.point_list) ) )
                    self.number_of_accute_angles=0
                    self.accute_angle_cleared=True
            
            if self.margin_left   > e.x: self.margin_left   = e.x 
            if self.margin_rigth  < e.x: self.margin_rigth  = e.x 
            if self.margin_top    > e.y: self.margin_top    = e.y 
            if self.margin_bottom < e.y: self.margin_bottom = e.y

            x_length,y_length,ratio = self.get_ratio(self.margin_left,self.margin_rigth,self.margin_top,self.margin_bottom)
            
            if (ratio >= self.RATIO_TRIGGER or self.minimun_ratio_trigered == True) \
                and self.accute_angle_cleared:
                rectangle_color = "yellow"
                self.minimun_ratio_trigered=True
                self.process_roi(y_length,x_length,self.margin_top,self.margin_left)
            else: 
                rectangle_color = "black"
            
            if self.rectangle: self.c.delete(self.rectangle)
            self.rectangle = self.c.create_rectangle(self.margin_left - RECTANGLE_MARGIN ,self.margin_bottom + RECTANGLE_MARGIN \
                                                     ,self.margin_rigth + RECTANGLE_MARGIN ,self.margin_top - RECTANGLE_MARGIN \
                                                     ,outline= rectangle_color)
        
        else:
            self.margin_left = e.x
            self.margin_rigth = e.x
            self.margin_top = e.y
            self.margin_bottom = e.y    

        self.old_x = e.x
        self.old_y = e.y
    
    def process_roi(self,y_length,x_length,y_start,x_start): 
        #print(y_length,x_length,y_start,x_start)
        MARGIN = 100
        HORIZONTAL_OFFSET = x_start  - int(MARGIN/2)
        VERTICAL_OFFSET = y_start  - int(MARGIN/2)
        
        self.roi_image = np.zeros((y_length + MARGIN,x_length + MARGIN),dtype='uint8')
        
        for i in range(len(self.point_list)-2):
            line_init = (self.point_list[i].x - HORIZONTAL_OFFSET ,self.point_list[i].y - VERTICAL_OFFSET )
            line_end =(self.point_list[i + 1].x - HORIZONTAL_OFFSET,self.point_list[i + 1].y - VERTICAL_OFFSET)
            cv2.line(self.roi_image, line_init, line_end, 255, 5)
        
        self.scaled_image = cv2.resize(self.roi_image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        
        
    def get_ratio(self, _margin_left,_margin_rigth,_margin_top,_margin_bottom):
        lenght_horizontal = self.margin_rigth - self.margin_left
        lenght_vertical = self.margin_bottom - self.margin_top
        horizontal_to_vertical_ratio =  lenght_vertical / lenght_horizontal
        #print("Lenght X:", lenght_horizontal, "Lenght Y:", lenght_vertical, "Ratio:", horizontal_to_vertical_ratio)
        return lenght_horizontal, lenght_vertical, horizontal_to_vertical_ratio      

    def clear_path(self, point_list_index):
        self.c.delete(ALL)
        #highlight
        if self.highlight_or_clear.get() == 1:
            for i in range(point_list_index - 2):
                self.c.create_line(self.point_list[i].x ,self.point_list[i].y,self.point_list[i+1].x,self.point_list[i+1].y,width=self.penwidth,fill='red',capstyle=ROUND,smooth=True)
        #clear
        if self.highlight_or_clear.get() == 2:
            del self.point_list[0:(point_list_index - 1)]
            self.margin_left    = self.old_x
            self.margin_rigth   = self.old_x
            self.margin_top     = self.old_y
            self.margin_bottom  = self.old_y

    def get_angle(self, point_start, point_middle, point_end):
        ang1 = np.arctan2(point_end.y - point_middle.y, point_end.x - point_middle.x) 
        ang2 = np.arctan2(point_start.y - point_middle.y, point_start.x - point_middle.x)
        angle = abs((ang1 - ang2) * (180 / np.pi))
        angle = round(angle)
        return angle
        

    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None  
        self.current_angle=None  
        self.number_of_accute_angles=0
        self.accute_angle_cleared=False
        self.minimun_ratio_trigered=False  

    def changeW(self,e): #change Width of pen through slider
        self.penwidth = e
           

    def clear(self,event=None):
        self.point_list.clear()
        self.c.delete(ALL)
        self.reset(self)

    def plot_roi(self):
        plt.imshow(self.scaled_image)
        plt.show() 
        _image=PIL.Image.fromarray(self.scaled_image)
        img =  PIL.ImageTk.PhotoImage(_image)
        self.mnist_output.configure(image=img)  

    def change_fg(self):  #changing the pen color
        self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  #changing the background color canvas
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg

    def change_drawing_mode(self):  # change the drawing mode, mouse movement with no button (continous drawing) 
                                    # or B1 pressed     
        if self.check_live.get() == True:
            
            self.c.unbind('<B1-Motion>')
            self.c.bind('<Motion>',self.paint)
        else:
            self.reset(self)
            self.c.unbind('<Motion>')
            self.c.bind('<B1-Motion>',self.paint)    

    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Pen Width:',font=('arial 18')).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_= 1, to = 50,command=self.changeW,orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1,ipadx=30)
      
        self.btnClear = ttk.Button(self.controls, text="CLEAR", command=self.clear).grid(row=1,column=0,padx=20, pady=0)
        self.btn_plt_roi = ttk.Button(self.controls, text="Plot ROI", command=self.plot_roi).grid(row=1,column=1,padx=20, pady=0)
        
        self.chkLive = Checkbutton(self.controls, text = "live drawing", command=self.change_drawing_mode, variable = self.check_live, height=5,width = 20).grid(row=1,column=2)
        
        Label(self.controls, text='On Angle Action:',font=('arial 14')).grid(row=2,column=0)

        self.r1=Radiobutton(self.controls, text="highlight", variable=self.highlight_or_clear,value=1).grid(row=2,column=1)
        self.r2=Radiobutton(self.controls, text="clear", variable=self.highlight_or_clear,value=2).grid(row=2,column=2)
        
        Label(self.controls, text='MNIST Output:',font=('arial 14')).grid(row=3,column=0)
        self.mnist_output = Label(self.controls,width=28,height=28,bg=self.color_bg)
        self.mnist_output.grid(row=3,column=1)
        
        self.controls.pack(side=LEFT)
        
        
        self.c = Canvas(self.master,width=640,height=480,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)
        
        self.master.bind('c', self.clear)

        self. change_drawing_mode()

        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('MNIST drawing test')
    root.mainloop()