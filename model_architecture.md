# INTENDED ARCHITECTURE FOR THE PROBLEM # 
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

conv → ReLU → pool
conv → ReLU → pool
conv → ReLU → pool
conv → ReLU → pool
flatten
linear → ReLU
linear → ReLU
linear (the output)

## Mathematical Breakdown ## 

# Hout = floor[(Hin + 2(P) - K) / S)] + 1
where: 
Hout = the height of the output image (and the width since we are using square images)
Hin = the input height of the image (structure uses a cascading structure therefore the input of one is the output of the last)
P = Padding value - amount of -infinity values added around the image to allow the kernel to move across everything 
K = Kernel size (2 = 2x2 kernel), the window we slide over the image, used in pooling to get the maximum value in a window which will make the image smaller 
S = Stride, the number of pixels the kernel moves in each step 
Floor function is used in case of any non-integer values 


| Type | $k$ | $s$ | $p$ |
| :--- | :-: | :-: | :-: |
| conv | 3   | 1   | 1   |
| pool | 2   | 2   | 0   |
