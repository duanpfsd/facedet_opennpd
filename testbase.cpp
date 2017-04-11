#include <iostream>
#include <fstream>
#include <io.h>
#include <string>
#include <vector>
#include <algorithm>
#include "conio.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
//#include "math_functions.h"
#include "npd/npddetect.h"
// include face.h here

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  npd::npddetect npd;
  npd.load("models/frontal_face_detector.bin");

  std::string path = "E:\\MyDownloads\\Download\\fktpxzq\\pic\\ztest\\testspeed\\";
  int showimg = 1;

  string pathName;
  long hFile = 0;
  struct _finddata_t fileInfo;
  int ncount = 0;

  std::ofstream infoout;
  infoout.open("./infoout/opencv.txt", ofstream::out);

  double t = cvGetTickCount();
  // Read picture files and store Face Information.
  if ((hFile = _findfirst(pathName.assign(path).append("/*").c_str(), &fileInfo)) == -1)
  {
    return -1;
  }
  while (_findnext(hFile, &fileInfo) == 0)
  {
    if (++ncount % 20 == 0)
    {
      cout << ncount << endl;
    }
    string imgname = pathName.assign(path).append(fileInfo.name).c_str();
    cv::Mat img_color = cv::imread(imgname, 1);
    if (img_color.empty())
    {
      //fprintf(stderr,"empty.\n");
      infoout << 0 << "  empty." << endl;
      continue;
    }
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);

    /*
      todo detect and landmark
      a list of int number is given, which indicate faces in each imagine.
    */

    int n = npd.detect(img_gray.data, img_gray.cols, img_gray.rows);
    vector< float > scores = npd.getScores();
    infoout << n << "  good  ";
    cout << n << "  good  ";
    /*
    for (int i = 0; i < n; ++i)
    {
      infoout << scores[i] << " ";
      cout << scores[i] << " ";
    }
    */
    infoout << endl;
    cout << endl;

    if (showimg)
    {
      float score = 0.1;
      vector< int >& Xs = npd.getXs();
      vector< int >& Ys = npd.getYs();
      vector< int >& Ss = npd.getSs();
      vector< float >& Scores = npd.getScores();
      char buf[10];
      for (int i = 0; i < n; i++)
      {
        if (score > 0. && Scores[i] < score)
          continue;
        sprintf(buf, "%.3f", Scores[i]);
        cv::rectangle(img_gray, cv::Rect(Xs[i], Ys[i], Ss[i], Ss[i]),
          cv::Scalar(128, 128, 128));
        cv::putText(img_gray, buf, cv::Point(Xs[i], Ys[i]), 1, 0.5, cv::Scalar(255, 255, 255));
      }
      imshow("Results_frontal", img_gray);
      cvWaitKey(0);
    }
  }
  t = cvGetTickCount() - t;
  cout << "Face detection and landmark consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
  infoout.close();
  _getch();

  return 0;
}


