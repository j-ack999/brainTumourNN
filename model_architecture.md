# Intended Architecture for the Problem # 
```text
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
```

```text
conv1 → ReLU → pool1
conv2 → ReLU → pool2
conv3 → ReLU → pool3
conv4 → ReLU → pool4
flatten
linear → ReLU
linear → ReLU
linear (the output)
```

# **Mathematical Breakdown** #

| Type | $k$ | $s$ | $p$ |
| :--- | :-: | :-: | :-: |
| conv | 3   | 1   | 1   |
| pool | 2   | 2   | 0   |


## Output size formulas

General form, for height and width (dilation 1):

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

where $k$ is the kernel (window) size, $s$ the stride and $p$ the padding on each side.

### Conv layer ($k=3$, $s=1$, $p=1$)

$$H_{out} = \left\lfloor \frac{H_{in} + 2(1) - 3}{1} \right\rfloor + 1 = H_{in}$$


### Pool layer ($k=2$, $s=2$, $p=0$)

$$H_{out} = \left\lfloor \frac{H_{in} + 2(0) - 2}{2} \right\rfloor + 1 = \left\lfloor \frac{H_{in}}{2} \right\rfloor$$

### conv1

$$H_{out} = \left\lfloor \frac{256 + 2(1) - 3}{1} \right\rfloor + 1 = 256$$

### pool1

$$H_{out} = \left\lfloor \frac{256 + 2(0) - 2}{2} \right\rfloor + 1 = 128$$




### *Further Model Architecture*


| Layer | $Input$ | $Output$ | $Comments$
| :--- | :-: | :-: | :-:
| conv1 | 3x256x256   | 16x256x256   |RGB therefore the in_channel is 3, with 16 filters therefore the out_channel is 16
| pool1 | 16x256x256   | 16x128x128| in_channel = 16 (unchanged)
| conv2 | 16x128x128   | 32x128x128   | out_channel = 32
| pool2 | 32x128x128   | 32x64x64 |in_channel = 32 (unchanged)
| conv3 | 32x64x64   | 64x64x64   |out_channel = 64
| pool3 | 64x64x64| 64x32x32|in_channel = 64 (unchanged)
| conv4 | 64x32x32| 128x32x32   |out_channel = 128
| pool4 | 128x32x32  | 128x16x16|in_channel = 128 (unchanged)
| flatten | 128x32x32  | 32,768|flatten the image so it's 1-dimensional and no longer 2d  


   


