/**
 * File: Matches.h
 * Project: DVision library
 * Author: Dorian Galvez-Lopez
 * Date: October 4, 2010
 * Description: Function to manage correspondences
 * License: see the LICENSE.txt file
 *
 */
 
#ifndef __D_MATCHES__
#define __D_MATCHES__
#if CV_NONFREE //Surf is non-free and may not be available
#include <vector>
#include <string>
#include "SurfSet.h"

namespace DVision {

/// Manages files of corresponding points
class Matches
{
public:

  /**
   * Saves two correspondence vectors in filename, w/ keypoints
   * @param filename
   * @param s1 surfset with keypoints from first image
   * @param s2 surfset with keypoints from second image
   * @param c1
   * @param c2 must be as long as c1
   */
  static void Save(const std::string &filename,
    const SurfSet &s1, const SurfSet &s2,
    const std::vector<int> &c1, const std::vector<int> &c2);

  /**
   * Loads two correspondence vectors from filename, w/ keypoints
   * @param filename
   * @param s1
   * @param s2
   * @param c1
   * @param c2 must be as long as c1
   */
  static void Load(const std::string &filename,
    SurfSet &s1, SurfSet &s2,
    std::vector<int> &c1, std::vector<int> &c2);

  /**
   * Saves two correspondence vectors in filename, w/o keypoints
   * @param filename
   * @param c1
   * @param c2 must be as long as c1
   */
  static void Save(const std::string &filename,
    const std::vector<int> &c1, const std::vector<int> &c2);
  
  /**
   * Loads two correspondence vectors from filename, w/o keypoints
   * @param filename
   * @param c1
   * @param c2
   */
  static void Load(const std::string &filename,
    std::vector<int> &c1, std::vector<int> &c2);
  
  /**
   * Loads two correspondence vectors from filename, w/o keypoints
   * @param filename
   * @param c1
   * @param c2
   */
  static void Load(const std::string &filename,
    std::vector<unsigned int> &c1, std::vector<unsigned int> &c2);

protected:
  
  /** 
   * Adss the correspondence vectors to the file storage
   * @param fs
   * @param c1
   * @param c2
   */
  static void save(cv::FileStorage &fs, const std::vector<int> &c1, 
    const std::vector<int> &c2);

  /** 
   * Loads the correspondence vectors from the file storage
   * @param fs
   * @param c1
   * @param c2
   */
  static void load(cv::FileStorage &fs, std::vector<int> &c1, 
    std::vector<int> &c2);

};

}

#endif
#endif
