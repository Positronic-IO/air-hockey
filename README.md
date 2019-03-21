# Air Hockey Vision and Robot Interface

## Introduction

In our goal to create an AI powered air hockey opponent, we have developed a vision system, state machine, and robot interface so that the robot can play in real time against a human opponent.

## Prerequisites

First, you will need to install all necessary dependencies.

### Python dependencies

Run `pip3 install pipenv`. `pipenv` is used for Python package management. 

Once downloaded, run `pipenv install`. This will create a virtual environment for our application and will install the dependencies from the `Pipfile.lock` file. If this is not present, it will default to the `Pipfile` file.

### YOLO and Darknet dependencies

We use YOLO2 and Darknet for the vision system. You will need to install `CUDA 9.0` and `cuDNN 7.0`. I would follow this [tutorial](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e) to set this up on your machine.

### Install Redis

You can either install redis via Docker or `apt-get`. It is up to your preference.

## Calibration and get up and running.

There are a few steps you have to do to sync up the camera footage with the robot.

The html canvas and robot sets the origin point at the top left corner. However, the top left corner of the table in the footage is not the origin. Thus, you will need to compute the offset necessary to translate the frames up and left so that the top left corner of the table in the footage is the origin point.

It does not matter if Redis has already been started or not.

_Make sure you are in your virtual environment. Run `pipenv shell` to initiate you environment if you are not already in it._

1. Grab a image from the source where your bot will be playing. This will be used to determine its location and dimensions.
2. Run `python calibrate_offset.py --file <Your Image>`. This will grab the necessary information and publish it to redis to be used for later.
3. In `retrieve_state.py`, set your defaults. The table dimensions either user-defined. Other defaults are dependent on your table and specifications. Please play with these.
4. Run `retrieve_state.py`. This will we retrieve the state of the entire application. It will also compute where the robot should move to. All puck/bot physics or AI should be defined in the `meet_the_puck` function.
5. In the `./simulator`, edit the `index.html` canvas components `width` and `height` attributes. Set these to the table dimensions.
6. In the same directory, run `npm start` to start simulator. Open `http://localhost:3000` to see where the puck and bot is on the table.
7. Run `python yolo/table.py` to start recording the state of the game from the YOLO2/Darknet neural net.


#### Author

Positronic Ai - Visit us at our [website](https://positronic.ai)! 