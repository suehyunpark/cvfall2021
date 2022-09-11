# M1522.001000 Computer Vision

This is a repository of my assignment submissions and project materials for [M1522.001000 Computer Vision](https://sugang.snu.ac.kr/sugang/cc/cc103.action?openSchyy=2021&openShtmFg=U000200002&openDetaShtmFg=U000300001&sbjtCd=M1522.001000&ltNo=001&sbjtSubhCd=000&lang_knd=ko) (Fall 2021 class, Seoul National University)
*Note: All of my solutions were uploaded well after the deadline of each assignment.*

## Project
### Visual Detection of Guitar Fretboard and Subsequent Finger Position Overlay By Augmenting Computer Vision Methodologies with Deep Learning
[Presentation slides](https://docs.google.com/presentation/d/1MarnvmWeA-f02vdsuZ9UmuCXFUcQdFh4l6AqUcq_0WE/edit?usp=sharing), [Report](https://github.com/suehyunpark/CV2021F/blob/master/project/%5BCV%20Project%20Final%20Report%5D%20Visual%20Detection%20of%20Guitar%20Fretboard%20and%20Subsequent%20Finger%20Position%20Overlay%20By%20Augmenting%20Computer%20Vision%20Methodologies%20with%20Deep%20Learning.pdf)


## hw1
* Convolution using reflect padding
* Median and Gaussian filter
* Gaussian and Laplacian pyramid
* Image blending using a laplacian pyramid
* Fourier transform (theory)


## hw2
* Convolution using replication padding
* Edge detection (Canny Edge Detector)
  * Non-maximum suppression (@ two neighboring pixels along the gradient direction)
  * Double thresholding
  * Edge tracking by hysteresis
* Hough transform for line detection
  * Non-maximum suppression (@ 8 neighboring pixels and the pixel itself)
* Hough line segments which do not extend beyond object boundaries
  * Detect initial line segments by storing Hough line pixels whose image magnitude is 1 (max value)
  * Merge short line segments
  * Find the longest segment identified inside each Hough line

*Results (image magnitude, Hough transform accumulator, Hough lines, Hough line segments) for each input image are stored in separate directories* <br>
*Detailed code explanation and result images are attached to `writeup.pdf`*


## hw3
* Camera model
* Camera calibration
* Homography
  * Normalize for stability (origin is the centroid of set of points)
* Image warp/mosaicing
  * Find correspondences through manual clicks
  * Bilinear interpolation
  * Warp and attach input image with respect to the reference image's perspective
* Image rectification
  * Similar to the `warp_image` function

*Detailed code explanation and result images are attached to `writeup.pdf`*

## hw4
* Epipolar geometry
* Optical flow
* Dominant motion estimation using Lucas-Kanade method
  * Warp image using the affine transformation matrix
  * Normalize the warped image gradients ∇I for calculation stability
* Moving object detection using affine motion subtraction
  * Iteratively update `p`, the optimal parameters to derive minimum SSD, until the search window moves by less than `epsilon`
  * Use double hysteresis thresholding to identify locations where the absolute difference of pixels in the template and the warped image domain exceeds the lower threshold and is below the upper threshold

*Detailed code explanation is attached to `writeup.pdf`*

## hw5
* K Nearest Neighbor
* Naive Bayes
  * Calculate likelihood using Gaussian density of a data point
  * Compute posterior probability given prior and likelihood
* Linear SVM
* Precision-Recall Curve
