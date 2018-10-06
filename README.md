Quant Motion System
===========================

## Usage

This system looks for a file in the home directory called "name". Its contents will be appended to all data to help with data organization of the scientist.

    $ echo 'name_of_device' > name

In the Quant_Mo directory, make sure you have the opencv packages installed, modify the camera system if needed (this system uses a raspberrypi with Adafruit camera)

    $ pip install -r requirements.txt

To run one pass of the motion collection system:

    $ bash whole_process experiment_name

* Note -- to gain any real insight you'll want to loop this^. In my experiment, I ran this command if generic motion was detected in the area.

To run motion analysis:

    $ ls directory_you_want_to_study | python analysis.py

This command^ will spit out a compressed system of analysis measurements and can even graph your results if you have matplotlib and 3d plotting utilities

## About

This system is a comprehensive guide for collecting and analyzing motion data in a systematic way. The purpose of this work is as an architectural tool for assesment of human movement patterns. This particular example is a proof of concept, using a raspberrypi with a camera as a recording instrument and a microwave diffraction sensor for broad motion filtering.

## Example

## Running this sample
For Mac/PC, simply run main.py.

    $ python main.py

For Raspberry Pi, run raspi_main.py.

    $ python raspi_main.py

## About code
=======
# Quant_Mo
