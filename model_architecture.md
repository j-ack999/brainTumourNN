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

Height and width are unchanged. Channels: $C_{out}$ = number of filters.

### Pool layer ($k=2$, $s=2$, $p=0$)

$$H_{out} = \left\lfloor \frac{H_{in} + 2(0) - 2}{2} \right\rfloor + 1 = \left\lfloor \frac{H_{in}}{2} \right\rfloor$$

Height and width are halved. Channels: $C_{out} = C_{in}$ (unchanged).
### conv1

$$H_{out} = \left\lfloor \frac{256 + 2(1) - 3}{1} \right\rfloor + 1 = 256$$

### pool1

$$H_{out} = \left\lfloor \frac{256 + 2(0) - 2}{2} \right\rfloor + 1 = 128$$










