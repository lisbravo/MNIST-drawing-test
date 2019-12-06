This small project is a simple and fast way to emulate a hand drawing digits in the air

Its main purpose is to analyze a good approach to solve the air hand writing digit calculator, called writing_dynamic_gestures_calculator, part of my Hand Commander framework

Its based on a sketchpad from https://github.com/abhishek305/ProgrammingKnowlegde-Tkinter-Series/blob/master/10th/Slider%20and%20color%20choos.py
that was modified to allow continuous live drawing (no need to hold a mouse button)  simulating the constant "drawing" path of hand movements

It also process the drawing to detect a valid digit and use a neural network to infer the current digit.

The choosed neural network is from the MNIST Keras example:  https://keras.io/examples/mnist_cnn/
This is the convolutional version which achieves an average 99.25% accuracy in a couple of minutes


The main challenges in this scenario was to "emulate" the start and stop conditions normally used in live handwriting
recognition, usually  pen down/up

Also, when drawing a new digit, the first movement is an ascending diagonal path to get to the point when a new digit begins,
since the camera will be registering your movements constantly, we need a way to detect such movement. 
This is a simple simulation on paper:

![On Paper](https://github.com/lisbravo/MNIST-drawing-test/raw/master/on_paper_sim.jpg)

Red highlights this first ascending path, Green the digit. 

So, after some tests, I decided that currently the fastest way to remove these initial movement was to detect the first acute angle between the first and the seccond drawed line as you can see highlighted in blue in the number one digit on previous image.
This has proven to be a pretty good solution as you can see:

![Starting line detection 1](https://github.com/lisbravo/MNIST-drawing-test/raw/master/starting_path_detection_1.gif)

And here you can see the code removing the initial move and also boxing the ROI (Region Of Interest) which is the base for further processing.   
The rectangle's color change to yellow when the ratio of horizontal to vertical sizes is over a preset level, a basic filter to detect a digit-like shape

![Starting line detection 2](https://github.com/lisbravo/MNIST-drawing-test/sraw/master/tarting_path_detection_2.gif)

Coincidently, this acute angle detection in combination with a proper ROI ratio trigger is also a good enough approach to detect the beginning of a digit drawing,
so, next in order is to detect a drawing finished condition. This has proven to be difficult and there are many approaches to be tested, 
but I decided for the simplest solution, to set a timer of x seconds from a start condition to allow the user to finish. After some tests,3 seconds seems to give enough time

From this point on, the code will grab the ROI, scale and normalize it accordingly to the neural network inputs and will infer the drawing.
There is threshold applied to the inference results to guarantee a good-enough recognition

Also is worth to mention that a watchdog timer is started after the first movement is detected, that will trigger if no valid input is detected

Even do there is a great margin for improvement, end result is pretty satisfactory:
![Results 1](https://github.com/lisbravo/MNIST-drawing-test/raw/master/end_result_1.gif)
![Results 2](https://github.com/lisbravo/MNIST-drawing-test/raw/master/end_result_2.gif)






