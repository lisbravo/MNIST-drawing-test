#based on: https://github.com/abhishek305/ProgrammingKnowlegde-Tkinter-Series/blob/master/10th/Slider%20and%20color%20choos.py
from tkinter import *
from tkinter import ttk, colorchooser
from collections import namedtuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageTk
import tensorflow as tf
from threading import Timer


class main:
    point = namedtuple('Point', ['x', 'y'])
    point_list = []
    RECTANGLE_MARGIN = 5
    
    #angle detection variables and CONSTANTS
    current_angle = None
    WINDOW_FOR_ANGLE_DETECTION = 15
    ANGLE_MIN_LIMIT = 140
    ANGLE_MAX_LIMIT = 220
    ACCUTE_ANGLE_TRIGGER = 3  # number of deteccted angles in the windows to trigger an "accute angle detected" routine
    number_of_accute_angles = 0
    accute_angle_cleared = False # indicates that the first, ascending line was detected and removed 

    #aspect ratio variables and constants
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

    #for inference
    TIME_TO_INFERENCE = 2 #this is the time to wait between a start condition (accute angle detected) and inference 
    WATCHDOG_TIME = 4 # To avoid blocking in the case that current drawing is too noisy  or gable
    PREDCITED_LABEL = 'Predicted #'
    PREDICTION_THRESHOLD = 0.7
    results=None
    inference_timer=None
    watchdog_timer=None
    # Graphic related 
    progress_bar_array = []
    progress_bar_val = []
    predictions_labels_array = []
    predictions_labels_var = []

    
    def __init__(self,master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.init_tensorflow()
        self.check_live = BooleanVar()
        self.check_live.set(True) 
        self.highlight_or_clear = IntVar(value=2)
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
        
        
    def init_tensorflow(self):
        # from: https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path='./MNIST_cnn_basic.tflite') 
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def paint(self,e):
        self.point_list.append(self.point(x = e.x,y = e.y))
        
        # first accute angle detection 
        if len(self.point_list) > self.WINDOW_FOR_ANGLE_DETECTION:
            mid_point = -1 - (round(self.WINDOW_FOR_ANGLE_DETECTION / 2))
            start_point = -1 - self.WINDOW_FOR_ANGLE_DETECTION
            self.current_angle = self.get_angle(self.point_list[start_point],self.point_list[mid_point],self.point_list[-1])
        
        if self.old_x and self.old_y:
            self.c.create_line(self.point_list[-2].x ,self.point_list[-2].y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

            #first line with accute angle removal
            if self.current_angle is not None \
                 and  (self.current_angle >= self.ANGLE_MAX_LIMIT \
                       or self.current_angle <= self.ANGLE_MIN_LIMIT):
                self.number_of_accute_angles +=1
                
                if self.number_of_accute_angles >= self.ACCUTE_ANGLE_TRIGGER \
                    and self.accute_angle_cleared == False:
                    self.clear_path( (len(self.point_list) ) )
                    self.number_of_accute_angles=0
                    self.accute_angle_cleared=True
            
            #setting the Region Of Interes     
            if self.margin_left   > e.x: self.margin_left   = e.x 
            if self.margin_rigth  < e.x: self.margin_rigth  = e.x 
            if self.margin_top    > e.y: self.margin_top    = e.y 
            if self.margin_bottom < e.y: self.margin_bottom = e.y

            #ratio is used to trigger the inference system, basically its a way to emulate a "pen up" event
            x_length,y_length,ratio = self.get_ratio(self.margin_left,self.margin_rigth,self.margin_top,self.margin_bottom)
            
            #at this point we have a valid input
            if (ratio >= self.RATIO_TRIGGER or self.minimun_ratio_trigered == True) \
                and self.accute_angle_cleared:
                rectangle_color = "yellow"
                self.minimun_ratio_trigered=True
                self.process_roi(y_length,x_length,self.margin_top,self.margin_left)
                # start inference timer and cancel the watchdog
                if (self.inference_timer is None) or \
                        (not self.inference_timer.is_alive()):
                    self.inference_timer = Timer(self.TIME_TO_INFERENCE, self.infer)
                    self.inference_timer.start()
                    self.watchdog_timer.cancel()
            else: 
                rectangle_color = "black"
            
            # drawing a rectangle to highlight the ROI
            if self.rectangle: self.c.delete(self.rectangle)
            self.rectangle = self.c.create_rectangle(self.margin_left - self.RECTANGLE_MARGIN ,self.margin_bottom + self.RECTANGLE_MARGIN \
                                                     ,self.margin_rigth + self.RECTANGLE_MARGIN ,self.margin_top - self.RECTANGLE_MARGIN \
                                                     ,outline= rectangle_color)
        
        else:
            # this is processed when it receives the firt point in a new "drawing"
            # the watchdog is set to clear an invalid input after timeup
            self.watchdog_timer = Timer(self.WATCHDOG_TIME,self.clear)
            self.watchdog_timer.start()
            self.margin_left = e.x
            self.margin_rigth = e.x
            self.margin_top = e.y
            self.margin_bottom = e.y    

        self.old_x = e.x
        self.old_y = e.y

    # refresh screen status with the new infered information
    def update_inference_results(self,event=None):
        prediction_string = None
        #block drawing input
        [self.c.unbind('<Motion>') if self.check_live.get() else self.c.unbind('<B1-Motion>') ]
        
        self.c.itemconfig(self.rectangle,outline='blue')
        
        if self.results[0][np.argmax(self.results[0])] >= self.PREDICTION_THRESHOLD:
            prediction_string = 'Number:' + str(np.argmax(self.results[0]))
        else: prediction_string ='Not Recognized'

        for i in range(len(self.results[0])):
            self.progress_bar_val[i].set(self.results[0][i])
            self.predictions_labels_var[i].set(str(int(self.results[0][i] * 100)) + "%")

        #locks any interaction, giving time to see the results
        self.c.create_text(self.margin_rigth + (self.RECTANGLE_MARGIN * 15),self.margin_top + self.RECTANGLE_MARGIN \
                            ,fill="darkblue",font="Times 20 italic bold", text= prediction_string)

        wait_timer = Timer(2.0,self.clear) 
        wait_timer.start() 

    # inference routine
    def infer(self,event=None):
        #normalization and reshaping 
        normalized_image = self.scaled_image.astype('float32') / 255 
        normalized_image = tf.reshape(normalized_image, [-1,28,28,1])
                
        self.interpreter.set_tensor(self.input_details[0]['index'], normalized_image)
        self.interpreter.invoke()
        self.results = self.interpreter.get_tensor(self.output_details[0]['index'])   
        
        self.update_inference_results(self)
        
                
    # this will process the valid Region Of Interest 
    def process_roi(self,y_length,x_length,y_start,x_start): 
        #print(y_length,x_length,y_start,x_start)
        MARGIN = 100
        HORIZONTAL_OFFSET = x_start  - int(MARGIN/2)
        VERTICAL_OFFSET = y_start  - int(MARGIN/2)
        THRESHOLD = 64
        
        # first it will create a new image with the ROI drawing, plus some margin for centering
        self.roi_image = np.zeros((y_length + MARGIN,x_length + MARGIN),dtype='uint8')
        
        for i in range(len(self.point_list)-2):
            line_init = (self.point_list[i].x - HORIZONTAL_OFFSET ,self.point_list[i].y - VERTICAL_OFFSET )
            line_end =(self.point_list[i + 1].x - HORIZONTAL_OFFSET,self.point_list[i + 1].y - VERTICAL_OFFSET)
            cv2.line(self.roi_image, line_init, line_end, 255, 8)
        # then it resizes the imagee to the 28*28 format of the neural model
        self.scaled_image = cv2.resize(self.roi_image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
         # applies a simple Threshold operation, improves inference
        self.scaled_image[self.scaled_image > THRESHOLD] = 255
        
        
    #simple routine to get the vertical to horizontal ratio, used to trigger a detection mechanism     
    def get_ratio(self, _margin_left,_margin_rigth,_margin_top,_margin_bottom):
        lenght_horizontal = self.margin_rigth - self.margin_left
        lenght_vertical = self.margin_bottom - self.margin_top
        horizontal_to_vertical_ratio =  lenght_vertical / lenght_horizontal
        #print("Lenght X:", lenght_horizontal, "Lenght Y:", lenght_vertical, "Ratio:", horizontal_to_vertical_ratio)
        return lenght_horizontal, lenght_vertical, horizontal_to_vertical_ratio      

    #eliminates the first line after an accute angle is detected
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
        
    #reset the whole system
    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None  
        self.current_angle=None  
        self.number_of_accute_angles=0
        self.accute_angle_cleared=False
        self.minimun_ratio_trigered=False
 
    #Clear Image
    def clear(self,event=None):
        self.point_list.clear()
        self.c.delete(ALL)
        self.reset(self)
        [pbar_val.set(0) for pbar_val in self.progress_bar_val]
        [label_var.set(0) for label_var in self.predictions_labels_var]
        self.change_drawing_mode()
        self.scaled_image =None
    
    def plot_roi(self,event=None):
        plt.imshow(self.scaled_image)
        plt.show() 

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
      
        self.btnClear = ttk.Button(self.controls, text="CLEAR", command=self.clear).grid(row=0,column=0,padx=20, pady=0)
        self.btn_plt_roi = ttk.Button(self.controls, text="Plot ROI", command=self.plot_roi).grid(row=0,column=1,padx=20, pady=0)
        
        self.chkLive = Checkbutton(self.controls, text = "live drawing", command=self.change_drawing_mode, variable = self.check_live, height=5,width = 20).grid(row=0,column=2)
        
        Label(self.controls, text='On Angle Action:',font=('arial 10')).grid(row=1,column=0)

        self.r1=Radiobutton(self.controls, text="highlight", variable=self.highlight_or_clear,value=1).grid(row=1,column=1)
        self.r2=Radiobutton(self.controls, text="clear", variable=self.highlight_or_clear,value=2).grid(row=1,column=2)
        
        row_base = 2
        
        for i in range(10):
            label_text = self.PREDCITED_LABEL + str(i) + ':'
            Label(self.controls, text=label_text,font=('arial 10')).grid(row=row_base + i,column=0)
            
            self.predictions_labels_var.append(StringVar())
            self.predictions_labels_var[i].set("%0")
            self.predictions_labels_array.append(\
                Label(self.controls, textvariable=self.predictions_labels_var[i],font=('arial 10')).grid(row=row_base + i,column=1)
                )
            
            self.progress_bar_val.append(DoubleVar())
            self.progress_bar_array.append(\
                    ttk.Progressbar(self.controls, orient = HORIZONTAL, length = 100, \
                                    variable= self.progress_bar_val[i], mode = 'determinate', maximum=1)) 
            self.progress_bar_array[i].grid(row=row_base + i,column=2)
            
        self.controls.pack(side=LEFT)
        
        self.c = Canvas(self.master,width=640,height=480,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)
        
        #shorcuts for testing on the fly
        self.master.bind('c', self.clear)
        self.master.bind('p', self.plot_roi)
        self.master.bind('i', self.infer)

        self. change_drawing_mode()

        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('MNIST drawing test')
    root.mainloop()



