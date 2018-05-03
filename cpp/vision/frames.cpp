#include "opencv2/opencv.hpp"
#include <time.h>
 
using namespace cv;
using namespace std;
 
int main(int argc, char** argv)
{
    Mat frame;              // Variable for storing video frames
    int num_frames = 120;   // Number of frames to capture
    time_t start, end;      // Start and end times

    // Start default camera
    VideoCapture cap(0);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "Error! Unable to open camera \n";
        return -1;
    }
     
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 340);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 220);
    cap.set(CV_CAP_PROP_FPS, 120);

    // With webcam get(CV_CAP_PROP_FPS) does not work.
    // Let's see for ourselves.
     
    // double fps = video.get(CV_CAP_PROP_FPS);
    // If you do not care about backward compatibility
    // You can use the following instead for OpenCV 3
    double fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;    
 
    cout << "Capturing " << num_frames << " frames" << endl ;
 
    // Start time
    time(&start);
     
    // Grab a few frames
    for(int i = 0; i < num_frames; i++)
    {
        cap >> frame;
    }
     
    // End Time
    time(&end);
     
    // Time elapsed
    double seconds = difftime (end, start);
    cout << "Time taken : " << seconds << " seconds" << endl;
     
    // Calculate frames per second
    fps  = num_frames / seconds;
    cout << "Estimated frames per second : " << fps << endl;
     
    // Release video
    cap.release();
    return 0;
}