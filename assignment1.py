import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy import signal
from scipy import optimize
from scipy.special import comb

data = pd.read_csv('sample.txt', sep = ' ')

#_______________ Data befrore filtering _______________
plt.figure()
plt.title("Data before filtering")
plt.plot(data['x'], data['y'], linewidth=0.2)

#_______________ Moving Average filter to reduce noise _______________
arr = data.to_numpy()
x, y = arr[:, 0], arr[:, 1]
window_len, exp_alpha = 101, 0.5
pad_left, pad_right = window_len // 2, (window_len - 1) // 2
y_padded = np.pad(y, (pad_left, pad_right), constant_values=(np.nan, np.nan))
exp_kernel_left = ((1 - exp_alpha) ** np.arange(1, pad_left + 1))[::-1]
exp_kernel_right = (1 - exp_alpha) ** np.arange(1, pad_right + 1)
exp_kernel = np.concatenate([exp_kernel_left, [1], exp_kernel_right])

avg_values = []
for i in range(len(y)):
    window = y_padded[i:i+window_len]
    exp_sum = exp_kernel[~np.isnan(window)].sum()
    exp_avg = np.nansum(window * exp_kernel) / exp_sum
    avg_values.append(exp_avg)

plt.figure()
plt.title("Data with moving average filter applied")
plt.plot(x, avg_values, lw=0.2)

#_______________ Savitzky-Golay filter as another quick alternative to reduce noise _______________
dataIIR = data
dataIIR['y'] = signal.savgol_filter(data['y'], 101, 2)

plt.figure()
plt.title("Data with Savitzky-Golay filter applied")
plt.plot(dataIIR['x'], dataIIR['y'], lw=0.2)

#_______________ Find location where steps occur _______________
# MA data
xs = np.array(x)
ys = np.array(avg_values)

# Finds step location and returns as array
diff = ys[1:] - ys[:-1]
indexBool = diff > 0.385 # Variable adjusted to fit number of steps
index = np.argwhere(indexBool).reshape(-1)
print(index)

# For plotting step
step1x = xs[(index[0]-100):(index[0]+100)]
step1y = ys[(index[0]-100):(index[0]+100)]
step2x = xs[(index[1]-100):(index[1]+100)]
step2y = ys[(index[1]-100):(index[1]+100)]
step3x = xs[(index[2]-100):(index[2]+100)]
step3y = ys[(index[2]-100):(index[2]+100)]

#_______________ Smoothing step  _______________

# To smooth transition between points - Doesn't connect end to rest of data
def sigmoid(x, mi, mx): 
    return mi + (mx-mi)*(lambda t: (1+200**(-t+0.5))**(-1) )( (x-mi)/(mx-mi) )

# Alternative to sigmoid junction - Clamps ends of sigmoid
def smoothclamp(x, mi, mx): 
    return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

# Possible future alternative where gradient of curve can be changed using N (ineffective - needs better vectorization)
def smoothstep(x, x_min=0, x_max=1, N=3):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    return result


# For plotting straight line between -100 and +100 of step
xl1 = np.linspace(xs[index[0]-100], xs[index[0]+100], 200) 
xl2 = np.linspace(xs[index[1]-100], xs[index[1]+100], 200) 
xl3 = np.linspace(xs[index[2]-100], xs[index[2]+100], 200) 
yl1 = np.linspace(ys[index[0]-100], ys[index[0]+100], 200) 
yl2 = np.linspace(ys[index[1]-100], ys[index[1]+100], 200) 
yl3 = np.linspace(ys[index[2]-100], ys[index[2]+100], 200)

# Applying smoothstep to y values of array
stepSmooth1 = smoothclamp(yl1, ys[index[0]-100], ys[index[0]+100])
stepSmooth2 = smoothclamp(yl2, ys[index[1]-100], ys[index[1]+100])
stepSmooth3 = smoothclamp(yl3, ys[index[2]-100], ys[index[2]+100])

# Highlight sections needed to be smoothed
plt.figure()
plt.title("Sections of data to be smoothed")
plt.plot(xs, ys, lw=0.2)
plt.plot(step1x, step1y)
plt.plot(step2x, step2y)
plt.plot(step3x, step3y)

# Highlight sections with smoothing compared to without
plt.figure()
plt.title("Sections of data before and after smoothstep function")
plt.plot(xs, ys, lw=0.2)
plt.plot(xl1, stepSmooth1)
plt.plot(xl2, stepSmooth2)
plt.plot(xl3, stepSmooth3)

# Apply smoothed curve data to MA
yf = ys
yf[index[0]-100:index[0]+100] = stepSmooth1
yf[index[1]-100:index[1]+100] = stepSmooth2
yf[index[2]-100:index[2]+100] = stepSmooth3

# Plot MA final
plt.figure()
plt.title("Data with moving average filter and smoothstep fuctions implemented")
plt.plot(xs, yf, lw=0.2)

plt.show()