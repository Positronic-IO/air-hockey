#./darknet detector test cfg/puck.data  cfg/yolo-puck.cfg weights/yolo-puck_4000.weights data/puck/310.jpg -i 0 -thresh 0.2
./darknet detector test cfg/two-pucks.data  cfg/yolo-two-pucks.cfg weights/yolo-two-pucks_300.weights data/two-pucks/1622.jpg -i 0 -thresh 0.5

