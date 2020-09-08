# VisiTel
## History
The Mitsubishi VisiTel was the finest webcam of 1988 listed for a mere $399 USD in [Popular Mechanics Feb 1988](https://books.google.com/books?id=HOQDAAAAMBAJ&lpg=PA50&ots=tzSafLrM-K&dq=visitel%20mitsubishi&pg=PA50#v=onepage&q=visitel%20mitsubishi&f=false)

## Run
* `pip install` the required dependancies 
* run `generate.py` to create images to send to a VisiTel
* run `decode.py` to decode an image recorded from a VisiTel device or specify `-l` to decode images captured live from the audio interface

## Protocol
The image is encoded as an AM signal ontop of a carrier at aproximately 1750Hz. The image is roughly 96 by 96 pixels in side but the drawn area may be a pixel or two smaller due to the CRT tuning.

## Hardware
To communicate with the VisiTel simply remove the RJ11 connector and replace it with a 3.5mm jack or otherwise interface it with a USB sound card. During each transmit or receive operation a relay will click. This is used to shut offf the handset (if connected) so that the user doesn't hear the image signal. If you don't hear this your device is not detecting your transmissions, try a different output volume. The Y joint in the VisiTel cable is a common point of failure for the audio signals.

## Issues
The demodulation does not a have a PLL as such it is tuned to the carrier frequency of my device I will attempt to improve this.

