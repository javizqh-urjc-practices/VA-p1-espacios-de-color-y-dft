/*
 Copyright (c) 2023 José Miguel Guerrero Hernández

 Licensed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License;
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     https://creativecommons.org/licenses/by-sa/4.0/

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "computer_vision/CVSubscriber.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#define PI 3.14159265

bool has_window = false;

typedef enum _filterMode {
  LOW_PASS_FILTER = 0,
  HIGH_PASS_FILTER
} filterMode;

// Create window at the beggining
void initWindow()
{
  if (has_window) return;
  has_window = true;

  // Show images in a different windows
  cv::namedWindow("window_name");
  // create Trackbar and add to a window
  cv::createTrackbar("Option [0-6]", "window_name", nullptr, 6, 0); 
  cv::createTrackbar("Filter Value [0-100]", "window_name", nullptr, 100, 0); 
}

// Convert from RGB to HSI
cv::Mat rgbToHSI(const cv::Mat & rgb_image)
{
  double r,g,b;
  double H,S,I;
  cv::Mat hsi(rgb_image.rows, rgb_image.cols, rgb_image.type());

  for (int i = 0; i < rgb_image.rows; i++) {
    for (int j = 0; j < rgb_image.cols; j++) {
      b = (double) rgb_image.at<cv::Vec3b>(i,j)[0];
      g = (double) rgb_image.at<cv::Vec3b>(i,j)[1];
      r = (double) rgb_image.at<cv::Vec3b>(i,j)[2];

      H = std::acos((((r-g)+(r-b))/2) / (std::sqrt((r-g)*(r-g) + (r-b)*(g-b))));
      S = 1 - ((3.0 * std::min(r, std::min(b,g)) / (r+g+b)));
      I = (r+g+b) / 3.0;

      if (b > g) H = 2*PI - H;

      hsi.at<cv::Vec3b>(i, j)[0] = (H * 180)/ PI ;
      hsi.at<cv::Vec3b>(i, j)[1] = S*255;
      hsi.at<cv::Vec3b>(i, j)[2] = I;
    }
  }

  return hsi;
}

cv::Mat createHorizFilter(const cv::Mat &image, const filterMode mode,
                          const int slider_val)
{
  float inside = (float) 0;  // Values inside the box
  float outside = (float) 0; // Values outside the box

  if (mode == LOW_PASS_FILTER) inside = 1;
  else if (mode == HIGH_PASS_FILTER) outside = 1;

  cv::Mat tmp(image.rows, image.cols, CV_32F);
  cv::Point center(image.rows / 2, image.cols / 2); // Is always even

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      if (i > center.x - slider_val && i < center.x + slider_val ) {
        tmp.at<float>(i, j) = inside;
      } else {
        tmp.at<float>(i, j) = outside;
      }
    }
  }

  cv::Mat toMerge[] = {tmp, tmp};
  cv::Mat horiz_Filter;
  cv::merge(toMerge, 2, horiz_Filter);

  return horiz_Filter;
}

// Compute the Discrete fourier transform
cv::Mat computeDFT(const cv::Mat & image)
{
  // Expand the image to an optimal size.
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(image.rows);
  int n = cv::getOptimalDFTSize(image.cols);     // on the border add zero values
  cv::copyMakeBorder(
    image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(
      0));

  // Make place for both the complex and the real values
  cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);           // Add to the expanded another plane with zeros

  // Make the Discrete Fourier Transform
  cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT);        // this way the result may fit in the source matrix
  return complexI;
}

// 6. Crop and rearrange
cv::Mat fftShift(const cv::Mat & magI)
{
  cv::Mat magI_copy = magI.clone();
  // crop the spectrum, if it has an odd number of rows or columns
  magI_copy = magI_copy(cv::Rect(0, 0, magI_copy.cols & -2, magI_copy.rows & -2));

  // rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = magI_copy.cols / 2;
  int cy = magI_copy.rows / 2;

  cv::Mat q0(magI_copy, cv::Rect(0, 0, cx, cy));     // Top-Left - Create a ROI per quadrant
  cv::Mat q1(magI_copy, cv::Rect(cx, 0, cx, cy));    // Top-Right
  cv::Mat q2(magI_copy, cv::Rect(0, cy, cx, cy));    // Bottom-Left
  cv::Mat q3(magI_copy, cv::Rect(cx, cy, cx, cy));   // Bottom-Right

  cv::Mat tmp;                             // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);                      // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);

  return magI_copy;
}

// Calculate dft spectrum
cv::Mat spectrum(const cv::Mat & complexI)
{
  cv::Mat complexImg = complexI.clone();
  // Shift quadrants
  cv::Mat shift_complex = fftShift(complexImg);

  // Transform the real and complex values to magnitude
  // compute the magnitude and switch to logarithmic scale
  // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
  cv::Mat planes_spectrum[2];
  cv::split(shift_complex, planes_spectrum);         // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
  cv::magnitude(planes_spectrum[0], planes_spectrum[1], planes_spectrum[0]);  // planes[0] = magnitude
  cv::Mat spectrum = planes_spectrum[0];

  // Switch to a logarithmic scale
  spectrum += cv::Scalar::all(1);
  cv::log(spectrum, spectrum);

  // Normalize
  cv::normalize(spectrum, spectrum, 0, 1, cv::NORM_MINMAX);   // Transform the matrix with float values into a
                                                      // viewable image form (float between values 0 and 1).
  return spectrum;
}
namespace computer_vision
{

/**
   TO-DO: Default - the output images and pointcloud are the same as the input
 */
CVGroup CVSubscriber::processing(
  const cv::Mat in_image_rgb,
  const cv::Mat in_image_depth,
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
const
{
  // Create output images
  cv::Mat out_image_rgb, out_image_depth;
  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  // Processing
  out_image_rgb = in_image_rgb;
  out_image_depth = in_image_depth;
  out_pointcloud = in_pointcloud;

  initWindow();
  switch (cv::getTrackbarPos("Option [0-6]", "window_name"))
  {
  case 0:
  {
    cv::imshow("window_name", out_image_rgb);
    break;
  }
  case 1:
  {
    cv::Mat HSI_image = rgbToHSI(in_image_rgb);
    cv::imshow("window_name", HSI_image);
    break;
  }
  case 2:
  {
    cv::Mat HSV_image;
    cv::Mat HSI_image = rgbToHSI(in_image_rgb);
    cv::cvtColor(out_image_rgb, HSV_image, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> three_channels_hsv;
    cv::split(HSV_image, three_channels_hsv );

    std::vector<cv::Mat> three_channels_hsi;
    cv::split(HSI_image, three_channels_hsi );

    cv::Mat result_h =  three_channels_hsv[0] - three_channels_hsi[0]; 
    cv::Mat result_s =  three_channels_hsv[1] - three_channels_hsi[1]; 
    cv::Mat result_iv = three_channels_hsv[2] - three_channels_hsi[2]; 

    std::vector<cv::Mat> final_channels;

    final_channels.push_back(result_h);
    final_channels.push_back(result_s);
    final_channels.push_back(result_iv);

    cv::Mat new_image;
    cv::merge(final_channels, new_image);

    cv::imshow("window_name", result_iv);;
    break;
  }
  case 3:
  {
    cv::Mat BW_opencv;
    cv::cvtColor(out_image_rgb, BW_opencv, cv::COLOR_RGB2GRAY);
    // Compute the Discrete fourier transform
    cv::Mat complexImg = computeDFT(BW_opencv);

    // Get the spectrum
    cv::Mat spectrum_original = spectrum(complexImg);

    cv::imshow("window_name", spectrum_original);
    break;
  }
  case 4:
  {
    cv::Mat BW_opencv;
    cv::cvtColor(out_image_rgb, BW_opencv, cv::COLOR_RGB2GRAY);
    // Compute the Discrete fourier transform
    cv::Mat complexImg = computeDFT(BW_opencv);

    // Crop and rearrange
    cv::Mat shift_complex = fftShift(complexImg); // Rearrange quadrants - Spectrum with low

    cv::Mat filter = createHorizFilter(in_image_rgb, LOW_PASS_FILTER,
                     cv::getTrackbarPos("Filter Value [0-100]", "window_name"));

    cv::mulSpectrums(shift_complex,filter,shift_complex,0);
    cv::Mat rearrange = fftShift(shift_complex);

    // Get the spectrum
    cv::Mat inverseTransform;
    cv::idft(rearrange, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);
    cv::imshow("window_name", inverseTransform);
    break;
  }
  case 5:
  {
    cv::Mat BW_opencv;
    cv::cvtColor(out_image_rgb, BW_opencv, cv::COLOR_RGB2GRAY);
    // Compute the Discrete fourier transform
    cv::Mat complexImg = computeDFT(BW_opencv);

    // Crop and rearrange
    cv::Mat shift_complex = fftShift(complexImg); // Rearrange quadrants - Spectrum with low

    cv::Mat filter = createHorizFilter(in_image_rgb, HIGH_PASS_FILTER,
                     cv::getTrackbarPos("Filter Value [0-100]", "window_name"));

    cv::mulSpectrums(shift_complex, filter, shift_complex,0);
    cv::Mat rearrange = fftShift(shift_complex);

    // Get the spectrum
    cv::Mat inverseTransform;
    //TODO: Esto en el ejercicio 6 cv::imshow("window_name", spectrum(rearrange));
    cv::idft(rearrange, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);
    cv::imshow("window_name", inverseTransform);
    break;
  }
  case 6:
  {
    cv::Mat BW_opencv;
    cv::cvtColor(out_image_rgb, BW_opencv, cv::COLOR_RGB2GRAY);
    // Compute the Discrete fourier transform
    cv::Mat complexImg = computeDFT(BW_opencv);
    cv::Mat complexImg_5 = complexImg.clone();

    // Crop and rearrange
    cv::Mat shift_complex_4 = fftShift(complexImg); // Rearrange quadrants - Spectrum with low
    cv::Mat shift_complex_5 = fftShift(complexImg_5); // Rearrange quadrants - Spectrum with low

    cv::Mat filter_4 = createHorizFilter(in_image_rgb, LOW_PASS_FILTER,
                     cv::getTrackbarPos("Filter Value [0-100]", "window_name"));

    cv::Mat filter_5 = createHorizFilter(in_image_rgb, HIGH_PASS_FILTER,
                     cv::getTrackbarPos("Filter Value [0-100]", "window_name"));

    cv::mulSpectrums(shift_complex_4,filter_4,shift_complex_4,0);
    cv::mulSpectrums(shift_complex_5,filter_5,shift_complex_5,0);
    cv::Mat rearrange_4 = fftShift(shift_complex_4);
    cv::Mat rearrange_5 = fftShift(shift_complex_5);

    // Get the spectrum
    cv::Mat inverseTransform_4;
    cv::idft(rearrange_4, inverseTransform_4, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inverseTransform_4, inverseTransform_4, 0, 1, cv::NORM_MINMAX);

    cv::Mat inverseTransform_5;
    cv::idft(rearrange_5, inverseTransform_5, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::normalize(inverseTransform_5, inverseTransform_5, 0, 1, cv::NORM_MINMAX);

    for (int i = 0; i < inverseTransform_4.rows; i++) {
      for (int j = 0; j < inverseTransform_4.cols; j++) {
        inverseTransform_4.at<float>(i,j) = inverseTransform_4.at<float>(i,j) > 0.6 ? 255 : 0;
      }
    }

    for (int i = 0; i < inverseTransform_5.rows; i++) {
      for (int j = 0; j < inverseTransform_5.cols; j++) {
        inverseTransform_5.at<float>(i,j) = inverseTransform_5.at<float>(i,j) > 0.4 ? 255 : 0;
      }
    }

    // OR
    cv::Mat Or_opencv;
    bitwise_or(inverseTransform_4, inverseTransform_5, Or_opencv);
    cv::imshow("window_name", Or_opencv);
    break;
  }
  default:
  {
    cv::imshow("window_name", out_image_rgb);
    break;
  }
  }
  cv::waitKey(3);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision
