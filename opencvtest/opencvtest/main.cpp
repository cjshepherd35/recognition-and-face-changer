//
//  main.cpp
//  opencvtest
//
//  Created by Colton Shepherd on 10/20/20.
//

#include <iostream>
#include <string>
//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

//file handling
#include <fstream>
#include <sstream>
//header files that contain functions
//#include "FaceRec.h"


using namespace std;
using namespace cv;
using namespace cv::face;


int main(int argc, const char * argv[]) {
//    // insert code here...
    
    namedWindow("face");
        Mat frame, gray, masksource;
        int i = 0, j = 1, testnumber = 50;
        string label;
        vector<int> labels;
        vector<Rect> faces;
        vector<Mat> facemats;
        string labelarray[2];

        bool m = true;
        VideoCapture cap(0);
        CascadeClassifier face_cascade;
        m -= (!face_cascade.load("/usr/local/Cellar/opencv/4.5.0/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml"));
        
        cout << "enter the name for  the first person \n";
        cin >> label;
        labelarray[0] = label;
        
        while (label !="-1") {
            while (m == true) {
                
                cap >> frame;
                m -= frame.empty();
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                equalizeHist(gray, gray);
                face_cascade.detectMultiScale(gray, faces, 1.2, 4, CASCADE_SCALE_IMAGE, Size(0,200));
                Mat facemat;
                Point leftcorner = Point(faces[0].tl().x, faces[0].tl().y);
                Point rightcorner = Point(faces[0].br().x, faces[0].br().y);
                rectangle(frame, leftcorner, rightcorner, Scalar(255,0,0));
                Mat ellipse_mask(gray(faces[0]).rows, gray(faces[0]).cols, CV_8UC1);
                ellipse_mask.setTo(0);
                cout << "faces size is " << faces.size() << endl;
                Point centerpoint = Point(ellipse_mask.rows/2, ellipse_mask.cols/2);
                ellipse(ellipse_mask, centerpoint, Size(faces[0].width*0.4,faces[0].height*2/3), 0, 0, 360, 255, -1);
                masksource = gray(faces[0]);
                masksource.copyTo(facemat, ellipse_mask);
                resize(facemat, facemat, Size(128,128));
                facemats.push_back(facemat);
                labels.push_back(j);
                
                
                imshow("face", facemats[i]);
                i+=faces.size();
                if (waitKey(100)>0 | i==(testnumber*j+testnumber)) {
                    break;
                }
            }
            cout << "enter the label number for another person \n";
            cout << "or -1 to exit. \n";
            cin >> label;
            labelarray[j] = label;
            j++;
        }
        
        Mat edgeface;
        m = true;
        Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();

        //train data
        model->train(facemats, labels);

        model->save("/Users/ColtonShepherd/Desktop/eigenfaces2.yml");

        cout << "Training finished...." << endl;
        while (m ==true) {
            cap >> frame;
            m -= frame.empty();
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            equalizeHist(gray, gray);
            face_cascade.detectMultiScale(gray, faces);
            
            for (int i = 0; i < faces.size(); i++) {
                Mat facemat;
                Point leftcorner = Point(faces[i].tl().x, faces[i].tl().y);
                Point rightcorner = Point(faces[i].br().x, faces[i].br().y);
                rectangle(frame, leftcorner, rightcorner, Scalar(255,0,0));
                Mat ellipse_mask(gray(faces[i]).rows, gray(faces[i]).cols, CV_8UC1);
                ellipse_mask.setTo(0);
                Point centerpoint = Point(ellipse_mask.rows/2, ellipse_mask.cols/2);
                ellipse(ellipse_mask, centerpoint, Size(faces[i].width*0.4,faces[i].height*2/3), 0, 0, 360, 255, -1);
                masksource = gray(faces[i]);
                masksource.copyTo(facemat, ellipse_mask);
                resize(facemat, facemat, Size(128,128));
                model->read("/Users/ColtonShepherd/Desktop/eigenfaces2.yml");
                int labelpredict = -1;
                double confidence = 0;
                model-> predict(facemat, labelpredict, confidence);
                cout << "label " << labelpredict << " confidence " << confidence << endl;
                if (labelpredict == 0) {
                    
                    putText(frame, labelarray[0], leftcorner, FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));
                    morphologyEx(frame(faces[i]), edgeface, MORPH_GRADIENT, Mat());                edgeface.convertTo(frame(faces[i]), CV_8UC1);
                    edgeface.convertTo(frame(faces[i]), CV_8U);
                }
                else if (labelpredict == 1) {
                    putText(frame, labelarray[1], leftcorner, FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));
                    frame(faces[i]) = 0;
                }
                else{
                    putText(frame, "unknown", leftcorner, FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0));
                }
            }
            
    //        cout << "label " << labelpredict << " confidence " << confidence << endl;
            imshow("face", frame);
            m -= (waitKey(10)>0);
        }
    
    return 0;
}
    
