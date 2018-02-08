/**
 * File: PMVSCamera.cpp
 * Project: DVision library
 * Author: Dorian Galvez-Lopez
 * Date: October 7, 2010
 * Description: Class to read camera info from PMVS output files
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <opencv2/core/core.hpp>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "PMVSCamera.h"
#include "DUtils.h"

using namespace std;
using namespace DVision::PMVS;

// ---------------------------------------------------------------------------

void CameraFile::readFile(const std::string &filename, Camera &cameras)
{
  fstream f(filename.c_str(), ios::in);
  
  string s;
  getline(f, s); // "CONTOUR"
  
  cv::Mat& P = cameras.P;
  P.create(3, 4, CV_64F);
  
  f >> P.at<double>(0,0) >> P.at<double>(0,1) >> P.at<double>(0,2) 
    >> P.at<double>(0,3)
    >> P.at<double>(1,0) >> P.at<double>(1,1) >> P.at<double>(1,2) 
    >> P.at<double>(1,3)
    >> P.at<double>(2,0) >> P.at<double>(2,1) >> P.at<double>(2,2) 
    >> P.at<double>(2,3);

  f.close();
}

// ---------------------------------------------------------------------------

void CameraFile::readFile(const std::string &filedir, 
  std::vector<Camera> &cameras)
{
  vector<string> files =
    DUtils::FileFunctions::Dir(filedir.c_str(), ".txt", true);
  
  cameras.resize(files.size());
  
  for(unsigned int i = 0; i < files.size(); ++i)
  {
    CameraFile::readFile(files[i], cameras[i]);
  }
}

// ---------------------------------------------------------------------------

void CameraFile::saveFile(const std::string &filename, const Camera &cameras)
{
  fstream f(filename.c_str(), ios::out);
  
  f << "CONTOUR" << endl;
  f.setf(ios::fixed, ios::floatfield);
  f.precision(6);

  const cv::Mat& P = cameras.P;
  
  f << P.at<double>(0,0) << " " << P.at<double>(0,1) << " "
    << P.at<double>(0,2) << " " << P.at<double>(0,3) << endl
    << P.at<double>(1,0) << " " << P.at<double>(1,1) << " "
    << P.at<double>(1,2) << " " << P.at<double>(1,3) << endl
    << P.at<double>(2,0) << " " << P.at<double>(2,1) << " "
    << P.at<double>(2,2) << " " << P.at<double>(2,3) << endl;

  f.close();
}

// ---------------------------------------------------------------------------

void CameraFile::saveFile(const std::string &filedir, 
  const std::vector<Camera> &cameras,
  const std::string& format)
{ 
  char filename[1024];

  for(unsigned int i = 0; i < cameras.size(); ++i)
  {
    sprintf(filename, format.c_str(), i);    
    CameraFile::saveFile(filedir + "/" + filename, cameras[i]);
  }
}

// ---------------------------------------------------------------------------


