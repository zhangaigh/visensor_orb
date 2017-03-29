/*!
 * \file FastCV_interface.h
 * \author  bo zhang <zhangbo24@lenovo.com>
 * \version 0.1
 * \date   03/08/2017
 * \section LICENSE
 *
 * \section DESCRIPTION
 *
 * This class is used to wrap fastcv function for pyramid computing and fast corner extracting. The class is invoked by ORBextractor class  
 * You can get pyramid images by getPyramidImage member function
 * You can get keypoints by getKeyPoints member function 

 */

#ifndef FAST_INTERFACE_H
#define FAST_INTERFACE_H

#include <malloc.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fastcv.h>
#include "dspCV.h"
#include "cornerApp.h"
#include "verify.h"
#include "msm_ion.h"
#include "fcntl.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

class FastCVExtractor
{
public:

    /// define vector data type for storing keypoints 
    typedef std::vector<std::vector<cv::KeyPoint>> vKeyPoints;     

    /// define variable related to dsp ion memory allocation
    int main_ion_fd;
    struct ion_allocation_data allocData;
    struct ion_fd_data ion_info_fd;


     /** \brief constructor .
     *
     *  
     */
    FastCVExtractor();
    ~FastCVExtractor();

    /** \brief init dsp setting and allocate memory .
     *
     *   @return  the status (0: success -1 failed)
     */
     int Init(int nWidth, int nHeight, int nFeatureNums, int nPyramidLevel, int nMaxThreshold , int nMinTreshold , int nBorder);


    /** \brief call wrapped fastcv interface to extractor fast corners .
     *   @param  img  input image 
     *   @return      the status (0: success -1 failed)
     */
    int StartExtractor(const cv::Mat& img);

protected:

     /** \brief release ion memory  .
     *
     *   @param ion       - memory pointer .
     *   @param size      - memory size
     */
    void IONMemFree(void *ion, uint32_t size);

    /** \brief allocate ion memory  .
     *
     *   @param size      - memory size
     */
    void * allocION(uint32_t size);
    
    /** \brief release memory and dsp status .
     *
     */
    void Release();

private:
    int m_nWidth;          ///<  image width
    int m_nHeight;         ///<  image height
    int m_nFeatureNum;     ///<  feature number 
    int m_nPyramidLevel;   ///<  pyramid level number
     cv::Mat m_mImg ;        ///<  input image

    int m_nMaxThreshold;   ///< maximum threshold for detecting fast corner
    int m_nMinThreshold;   ///< minimum threshold for detecting fast corner
     int m_nBorder;         ///< boarder 

    int m_nScoreNum;       ///< number corner score 



    const int scaledPatchSize = 31;    ///< descriptor patch size

     uint32_t* m_piCorners;    ///< pointer to detected corner 
     uint8_t* m_piDst;         ///< pointer to pyramid images 
     uint32_t* m_piNumCornersDetected;    ///< the number of detected corner in each pyramid level
     uint32_t* m_piScore;                 ///< the score pointer of corner


public:

     /** \brief fetch pyramid images .
     *   @return  pyramid image vector 
     */
     std::vector<cv::Mat> getPyramidImage();

    /** \brief fetch keypoints in all pyramid levels.
     *   @return  keypoints  
     */
     vKeyPoints getKeyPoints();
};

}

#endif 











