import numpy as np
import wavio
import cv2
import matplotlib.pyplot as plt
import sys
import pyaudio
import argparse

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
FREQ = 1750
FFT_SIZE = 64

class VisitelImage:

    LINE_LENGTH = 95.5
    IMAGE_SIZE_X = 95
    IMAGE_SIZE_Y = 90

    def __init__(self, samples_per_wave):
        self.samples_per_wave = samples_per_wave
        self.x = 0
        self.y = 0
        self.lastLine = 0
        self.img = np.zeros(
            shape=[VisitelImage.IMAGE_SIZE_Y, VisitelImage.IMAGE_SIZE_X, 1], dtype=np.uint8)

    def output_pixel(self, i, value):
        if(self.lastLine == 0):
            self.lastLine = i - (self.samples_per_wave)
        if(self.y < VisitelImage.IMAGE_SIZE_Y and self.x < VisitelImage.IMAGE_SIZE_X):
            self.img[self.y, self.x] = (1-value) * 255
            if(value < 0 or value > 1):
                print("error")
        self.x += 1
        if((i - self.lastLine) >= round((VisitelImage.LINE_LENGTH * self.samples_per_wave))):
            self.x = 0
            self.y += 1
            self.lastLine = i

    def finalize(self):
        img = self.img
        img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)
        big = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        big = cv2.resize(big, (VisitelImage.IMAGE_SIZE_Y*5, VisitelImage.IMAGE_SIZE_X*5),
                         interpolation=cv2.INTER_NEAREST)
        o = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
        outX = round((big.shape[1]/2-img.shape[1]/2)/2)
        o[0:big.shape[0], outX:outX+big.shape[1]] = big
        return o


class WavFile:

    def __init__(self, file_path):
        self.wav = wavio.read(file_path).data[:, 0]
        norm = max((abs(np.amax(self.wav)), (abs(np.min(self.wav)))))
        self.wav = self.wav / norm
        self.wav_pos = 0

    def get_data(self):
        if(self.wav_pos + CHUNK < len(self.wav)):
            ret = self.wav[self.wav_pos: self.wav_pos + CHUNK]
        else:
            ret = [0] * CHUNK
        self.wav_pos += CHUNK
        return ret

    def has_data(self):
        return self.wav_pos + CHUNK < len(self.wav)

class WavStream:
    def __init__(self):
        self.stream = pyaudio.PyAudio().open(format=FORMAT,
                                                 channels=CHANNELS,
                                                 rate=RATE,
                                                 input=True,
                                                 frames_per_buffer=CHUNK)
        self.max_amplitude = 1

    def get_data(self):
        data = self.stream.read(CHUNK)
        # import struct
        # struct.unpack(str(CHUNK)+'f',data)
        data = np.frombuffer(data, dtype=np.float32)
        return data

    def has_data(self):
        return True


class VisitelWebcam:

    WINDOW_NAME = 'Visitel Image'

    def __init__(self, live, wav_file_path, v4l2_device, headless):
        self.v4l2_device = v4l2_device
        self.live = live
        self.headless = headless

        self.spw = 25.23158  # 25.23
        self.input = []
        self.skip = 0
        self.reset()
        if(self.v4l2_device):
            import pyfakewebcam
            self.camera = pyfakewebcam.FakeWebcam(self.v4l2_device, 640, 480)
        if(self.live):
            self.wav = WavStream()
        else:
            self.wav = WavFile(wav_file_path)
        print("inited")

    def reset(self):
        self.state = 0
        self.previousSample = 0
        if(self.live):
            self.input = []
        self.i = 0
        self.peakIndex = 0
        self.foundPeak = False
        self.peakThisWave = False
        self.imagePosition = 0.0
        self.imageEnd = 0
        self.max_amplitude = 1
        self.img = VisitelImage(self.spw)

    def find_signal(self):
        while(self.i < len(self.input)-FFT_SIZE):
            fft = np.fft.fft(self.input[self.i:self.i + FFT_SIZE])
            fbin = round(len(fft)/2 * FREQ / RATE)
            if(fft[fbin] > 8):
                self.max_amplitude = max(
                    [abs(sample) for sample in self.input[self.i:self.i + FFT_SIZE]])
                return True
            self.i += FFT_SIZE
        return False

    def find_quienence(self):
        while(self.i < len(self.input)-FFT_SIZE):
            fft = np.fft.fft(self.input[self.i:self.i + FFT_SIZE])
            fbin = round(len(fft)/2 * FREQ / RATE)
            if(fft[fbin] < 2):
                return True
            self.i += FFT_SIZE
        return False

    def synthronize_osc(self):
        return 25.23158

    def read_image(self):
        while (self.i < len(self.input) and self.i < self.imageEnd):
            s = self.input[self.i]

            if(s < 0):
                self.peakThisWave = False

            # Fix a little bit of aliasing around around phase switches add `or s>self.input[peakIndex]):`
            if(s > 0 and (s < self.previousSample and self.peakThisWave == False)):
                self.peakThisWave = True
                self.peakIndex = min((self.i-1), len(self.input)-1)
                if (self.foundPeak == False):
                    self.foundPeak = True
                    self.imagePosition = self.peakIndex + self.spw

            if (((self.foundPeak == True and self.i == round(self.imagePosition)))):
                distanceFromPeak = abs(self.imagePosition - self.peakIndex)
                # Visitel shifts the carrier out of phase to indicate pure white pixels
                if(distanceFromPeak > 6 and distanceFromPeak < 20):     # out of phase
                    self.img.output_pixel(self.i, 0)
                else:
                    self.img.output_pixel(
                        self.i, abs(s) / self.max_amplitude)
                self.imagePosition += self.spw
            self.previousSample = s
            self.i += 1

    def main_loop(self):
        if(self.live == False):
            while self.wav.has_data():
                self.input.extend(self.wav.get_data())
        inputs = 0
        while(True):
            while(True):
                if(self.live == True):
                    inputs += 1
                    data = self.wav.get_data()
                    # normalize the stream
                    data = [datum * (1 / self.max_amplitude) for datum in data]
                    self.input += data
                    if(inputs > 1000):
                        self.i = 0
                        inputs = 0
                        self.input = []
                    print(f"inputs: {inputs} state: {self.state}")
                else:
                    print(f"state: {self.state}")
                if(self.state == 0):
                    self.state += 1 if self.find_signal() else 0
                elif(self.state == 1):
                    self.state += 1 if self.find_quienence() else 0
                elif(self.state == 2):
                    if(self.find_signal()):
                        self.i += round(RATE * (0.413 + self.skip))
                        self.state += 1
                        # video frame is 5 seconds
                        self.imageEnd = self.i + round(5*RATE)
                elif(self.state == 3):
                    self.spw = self.synthronize_osc()
                    self.state += 1
                elif(self.state == 4):
                    self.read_image()
                    if(self.i >= self.imageEnd or self.i >= len(self.input)):
                        state = 0
                        break

            output = self.img.finalize()
            if(self.v4l2_device):
                camera.schedule_frame(output)
            if(self.headless == False):
                cv2.imshow(VisitelWebcam.WINDOW_NAME, output)
                key = cv2.waitKey(100 if self.live else 0)
                if(key == 113):  # q
                    self.spw += 0.0001
                elif(key == 120 or cv2.getWindowProperty(VisitelWebcam.WINDOW_NAME, 0) < 0):  # x
                    break
                elif(key == 115):  # s
                    self.skip += 0.001
                    print(self.skip)
            self.reset()

parser = argparse.ArgumentParser(description='Decode visitel images')
parser.add_argument('-l', '--live',  action='store_true', help="Specify live to sample audio from the default audio input device")
parser.add_argument('-wp', '--wav_path', metavar='', action='store', default='input.wav', help="The path to a wav file to decode an image from")
parser.add_argument('-v','--v4l2_output', metavar='', action='store', help="Path to a v4l2 loopback device to emit frames into, LINUX ONLY")
parser.add_argument('-hl','--headless', action='store_true', help="Don't display the decoded image")
args = parser.parse_args()

VisitelWebcam(args.live, args.wav_path, args.v4l2_output, args.headless).main_loop()
