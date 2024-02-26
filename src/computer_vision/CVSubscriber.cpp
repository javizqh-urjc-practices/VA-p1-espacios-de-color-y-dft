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
#include "rclcpp/rclcpp.hpp"


// Compute the Discrete fourier transform
cv::Mat computeDFT(const cv::Mat &image)
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

inline float rgb2hue(const float r, const float g, const float b){
  return std::acos(
    ( ((r-g) + (r-b))/2 )/
    ( std::sqrt((std::pow(r-g,2) + (r-b)*(g-b))) )
  );

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

namespace CVParams {

  inline bool running = false;
  bool key_pressed = false;
  inline std::string WINDOW_NAME = "Practica_5";
  inline std::string WINDOW_A_NAME = "Umbral A (ex.6)";
  inline std::string WINDOW_A_SPEC_NAME = "Spectrum A (ex.6)";
  inline std::string WINDOW_B_NAME = "Umbral B (ex.6)";
  inline std::string WINDOW_B_SPEC_NAME = "Spectrum B (ex.6)";

  int MAX_STRENGH = 100;
  float UMBRAL_CASE_4 = 0.6;
  float UMBRAL_CASE_5 = 0.4;
  
  enum FILTER_PARAMS {
    VERTICAL,
    HORIZONTAL,
    INSIDE,
    OUTSIDE
  };

}
namespace CVFunctions {

std::vector<cv::Mat> multichannelDFT(const cv::Mat &image_rgb){

  // Declarations
  std::vector<cv::Mat> input_channels;
  std::vector<cv::Mat> output_channels;

  // Computing DFT for each
  cv::split(image_rgb, input_channels);
  for (int i = 0; i < (int)input_channels.size(); i++){
    output_channels.push_back(computeDFT(input_channels[i]));
  }

  return output_channels;
}



cv::Mat complex_channels2RGBspectrum(const std::vector<cv::Mat>& complex_channels)
{
    cv::Mat result;
    std::vector<cv::Mat> spectrums;

    for (int i = 0; i < (int)complex_channels.size(); ++i) {
        spectrums.push_back(spectrum(complex_channels[i]));
    }

    cv::merge(spectrums, result);
    return result;
}

cv::Mat BGR2HSI(const cv::Mat & image){

  cv::Mat hsi_mat = cv::Mat::zeros(image.size(), image.type());
  float r, g, b, h, s, i;

  for (int u = 0; u < image.rows; u++) {
    for (int v = 0; v < image.cols; v++) {

      b = ((float)image.at<cv::Vec3b>(u,v)[0]) / 255;
      g = ((float)image.at<cv::Vec3b>(u,v)[1]) / 255;
      r = ((float)image.at<cv::Vec3b>(u,v)[2]) / 255;

      h = rgb2hue(r,g,b);
      if (r + g + b != 0) {
        s = 1 - 3 * std::min({r, g, b}) / (r + g + b);
      }else{
        s = 0;
      }
      i = (r+g+b)/3;

      if (b > g) {
        h = 360 - h;
      }

      hsi_mat.at<cv::Vec3b>(u,v)[0] = h*255/360;
      hsi_mat.at<cv::Vec3b>(u,v)[1] = s*255;
      hsi_mat.at<cv::Vec3b>(u,v)[2] = i*255;
    }
  }

  return hsi_mat;

}

cv::Mat substract_channels(const cv::Mat & mat1, const cv::Mat & mat2) {

  std::vector<cv::Mat> channels1;
  std::vector<cv::Mat> channels2;
  std::vector<cv::Mat> splitted_result;

  cv::split(mat1, channels1);
  cv::split(mat2, channels2);

  for (int i = 0; i < (int)channels1.size(); i++) {
    splitted_result.push_back(channels1[i] - channels2[i]);
  }

  cv::Mat result;
  cv::merge(splitted_result,result);

  return result;

}

cv::Mat filter(const cv::Mat &image, float strength, CVParams::FILTER_PARAMS direction, CVParams::FILTER_PARAMS mode) {
    // Warning: Asumes complex images as input

    int start, end;
    cv::Mat result = cv::Mat::zeros(image.size(), image.type());

    strength = std::max(0.0f, std::min(1.0f, strength));

    int insideMultiplier = mode == CVParams::INSIDE ? 1 : 0;
    int outsideMultiplier = 1 - insideMultiplier;
    int target = direction == CVParams::VERTICAL ? image.cols : image.rows;

    int deleted = static_cast<int>(strength * (target / 2.0));
    start = std::max(target / 2 - deleted, 0);
    end = std::min(target / 2 + deleted, image.cols);

    int i, j = 0;
    int& line = direction == CVParams::VERTICAL ? j : i;

    for (i = 0; i < result.rows; i++) {
        for (j = 0; j < result.cols; j++) {
            if (line >= start && line < end) {
                result.at<cv::Vec2f>(i, j) = image.at<cv::Vec2f>(i, j) * insideMultiplier;
            } else {
                result.at<cv::Vec2f>(i, j) = image.at<cv::Vec2f>(i, j) * outsideMultiplier;
            }
        }
    }

    return result;
}

cv::Mat build_filter(const cv::Mat &image, float strength, CVParams::FILTER_PARAMS direction, CVParams::FILTER_PARAMS mode) {
    // Warning: Asumes complex images as input

    int start, end;
    cv::Mat result = cv::Mat::zeros(image.size(), image.type());

    strength = std::max(0.0f, std::min(1.0f, strength));

    float insideMultiplier = mode == CVParams::INSIDE ? 0.0f : 1.0f;
    float outsideMultiplier = 1.0f - insideMultiplier;

    int target = direction == CVParams::VERTICAL ? image.cols : image.rows;

    int deleted = static_cast<int>(strength * (target / 2.0));
    start = std::max(target / 2 - deleted, 0);
    end = std::min(target / 2 + deleted, image.cols);

    int i, j = 0;
    int& line = direction == CVParams::VERTICAL ? j : i;

    for (i = 0; i < result.rows; i++) {
        for (j = 0; j < result.cols; j++) {
            if (line >= start && line < end) {
                result.at<cv::Vec2f>(i, j) = cv::Vec2f(insideMultiplier, insideMultiplier);
            } else {
                result.at<cv::Vec2f>(i, j) = cv::Vec2f(outsideMultiplier, outsideMultiplier);
            }
        }
    }

    return result;
}

cv::Mat umbral(const cv::Mat& input, float ratio) {
  // Warning: Assumes 1 Channel input image

  cv::Mat result = cv::Mat::zeros(input.size(), input.type());

  for (int i = 0; i < result.rows; i++) {
      for (int j = 0; j < result.cols; j++) {
          result.at<float>(i, j) = input.at<float>(i,j) > ratio ? 255 : 0;
      }
  }

  return result;
}

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
  cv::Mat out_image_rgb, out_image_depth, preprocessed_image, hsv, hsi, complex_image, filtered_image, gray_image;
  cv::Mat image_ab, filter_a, filter_b, filter, filtered_complex_image, complex_a, complex_b;

  std::vector<cv::Mat> dft_channels;
  int strength;
  CVParams::FILTER_PARAMS mode, direction;
  cv::Mat spectrum_a = cv::Mat::zeros(in_image_rgb.size(), in_image_rgb.type());
  cv::Mat spectrum_b = cv::Mat::zeros(in_image_rgb.size(), in_image_rgb.type());
  cv::Mat image_a = cv::Mat::zeros(in_image_rgb.size(), in_image_rgb.type());
  cv::Mat image_b = cv::Mat::zeros(in_image_rgb.size(), in_image_rgb.type());

  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  // Processing
  out_image_rgb = in_image_rgb;
  out_image_depth = in_image_depth;
  out_pointcloud = in_pointcloud;

  // First time execution
  if (!CVParams::running) {

    CVParams::running = true;

    cv::namedWindow(CVParams::WINDOW_NAME);
    // create Trackbar and add to a window
    cv::createTrackbar("mode", CVParams::WINDOW_NAME, nullptr, 6, 0);
    cv::createTrackbar("filter_value", CVParams::WINDOW_NAME, nullptr, CVParams::MAX_STRENGH, 0);
  }

  // Obtaining Parameter
  strength = cv::getTrackbarPos("filter_value", CVParams::WINDOW_NAME);
    
  switch (cv::getTrackbarPos("mode", CVParams::WINDOW_NAME))
  {
  case 1:
    preprocessed_image = CVFunctions::BGR2HSI(in_image_rgb);
    break;

  case 2:

    cv::cvtColor(in_image_rgb, hsv, cv::COLOR_BGR2HSV);
    hsi = CVFunctions::BGR2HSI(in_image_rgb);
    preprocessed_image = CVFunctions::substract_channels(hsv, hsi);
    break;

  case 3:

    cv::cvtColor(in_image_rgb, preprocessed_image, cv::COLOR_RGB2GRAY);
    complex_image = computeDFT(preprocessed_image);
    preprocessed_image = spectrum(complex_image);
    break;

  case 4:
  case 5:


    if (cv::getTrackbarPos("mode", CVParams::WINDOW_NAME) == 4) {

      // Maintain only main horizontal 
      direction = CVParams::HORIZONTAL;
      mode = CVParams::OUTSIDE;

    } else {

      // Remove only main horizontal 
      direction = CVParams::HORIZONTAL;
      mode = CVParams::INSIDE;

    }

    // Changing to GrayScale
    cv::cvtColor(in_image_rgb, gray_image, cv::COLOR_RGB2GRAY);

    // Preprocessing
    complex_image = fftShift(computeDFT(gray_image));

    filter = CVFunctions::build_filter(complex_image, (float)strength/CVParams::MAX_STRENGH, direction, mode);
    cv::mulSpectrums(complex_image, filter, filtered_complex_image, 0);

    complex_image = fftShift(filtered_complex_image);
    cv::idft(complex_image, filtered_image, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    // Normalizing
    cv::normalize(filtered_image, preprocessed_image, 0, 1, cv::NORM_MINMAX);

    break;

  case 6:

    // Changing to GrayScale
    cv::cvtColor(in_image_rgb, gray_image, cv::COLOR_RGB2GRAY);
    complex_image = fftShift(computeDFT(gray_image));

     // Building Filters
    filter_a = CVFunctions::build_filter(complex_image, (float)strength/CVParams::MAX_STRENGH, CVParams::HORIZONTAL, CVParams::OUTSIDE);
    filter_b = CVFunctions::build_filter(complex_image, (float)strength/CVParams::MAX_STRENGH, CVParams::HORIZONTAL, CVParams::INSIDE);

    cv::mulSpectrums(complex_image, filter_a, complex_a, 0);
    cv::mulSpectrums(complex_image, filter_b, complex_b, 0);

    spectrum_a = spectrum(fftShift(complex_a));
    spectrum_b = spectrum(fftShift(complex_b));

    // Returning to bgr image
    cv::idft(fftShift(complex_a), image_a, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::idft(fftShift(complex_b), image_b, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    // Normalizing
    cv::normalize(image_a, image_a, 0, 1, cv::NORM_MINMAX);
    cv::normalize(image_b, image_b, 0, 1, cv::NORM_MINMAX);

    // Umbralizing
    image_a = CVFunctions::umbral(image_a,CVParams::UMBRAL_CASE_4);
    image_b = CVFunctions::umbral(image_b,CVParams::UMBRAL_CASE_5);

    // OR
    cv::bitwise_or(image_a, image_b, preprocessed_image);
    break;

  default:
    preprocessed_image = in_image_rgb;
    break;
  }

  if (cv::waitKey(5) == 'd') {
    CVParams::key_pressed = !CVParams::key_pressed;
  }

  if (CVParams::key_pressed) {
    cv::imshow(CVParams::WINDOW_A_NAME, image_a);
    cv::imshow(CVParams::WINDOW_B_NAME, image_b);
    cv::imshow(CVParams::WINDOW_A_SPEC_NAME, spectrum_a);
    cv::imshow(CVParams::WINDOW_B_SPEC_NAME, spectrum_b);
  }

  cv::imshow(CVParams::WINDOW_NAME, preprocessed_image);
  cv::waitKey(1);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision
