import numpy as np
import wavio
import cv2

# Load an color image in grayscale
img = cv2.imread('test.png',0)
img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

rate = 44100.0  # samples per second
f = 1764        # sound frequency (Hz)
T = 1 / f       # sample duration (seconds)
samples = T * rate

# Build a waveshape
#t = []
#for i in range(0,int(samples)):
#    t.append(i * T/samples)
#t = np.array(t)

t = np.linspace(0, T, samples, endpoint=False)
x = np.sin(2 * np.pi * f * t)   # make a nice sinewave
x = x * 0.6                     # attenuate it to match our header

def add_pixel(frame, pos, brightness):
    # t = np.linspace(0, T, samples, endpoint=False)
    brightness = (1- brightness)
    frame[pos:pos+len(x)] = x * brightness
    #return np.concatenate((frame, x * brightness))

def pad(frame, pos, pixels):
    extras = 22# round(((55.0/1000.0) * rate) - (pixels * samples)) 54.932
    frame[pos:pos+extras] = [0] * extras
    #return np.concatenate((frame,[0] * extras))

#T = 0.0545       # sample duration (seconds)
#f = 1750.0     # sound frequency (Hz)
#t = np.linspace(0, T, T*rate, endpoint=False)
#x = np.sin(2*np.pi * f * t)
#wavio.write("sine24.wav", x, rate, sampwidth=3)

def make_image():
    #xx= (x * 0.5)
    header = wavio.read("header.wav").data[:, 0]
    norm = max((abs(np.amax(header)),(abs(np.min(header))))) 
    header = header / norm
    z = np.zeros(261039) # length of header plus image pixels plus pads
    z[0:len(header)] = header
    pos = len(header)
    # add aprox 1 line of white to black and align with the drawn area of the screen
    for y in range (0,92):
        add_pixel(z, pos, (y/101))
        pos+=len(x)

    # write out the frame
    for i in range(0,96):
        pixels = 96
        for y in range (0,pixels):
            add_pixel(z, pos, img[i,min(y,99)] if y < 95 else 0)
            pos+=len(x)
        pad(z, pos, pixels)
        pos+=22
    print(len(z))
    wavio.write("output.wav", z, rate, sampwidth=3)

make_image()