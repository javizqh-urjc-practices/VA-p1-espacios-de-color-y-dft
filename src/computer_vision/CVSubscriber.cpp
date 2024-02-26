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


// -------- DFT Imported Functions ----------------------------------------------
// Copied because it is only allowed to edit this file

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


// -------- Parameters ----------------------------------------------


namespace CVParams {

  inline bool running = false;
  bool extra_active = false;

  inline std::string WINDOW_NAME = "Practica_5";
  inline std::string WINDOW_A_NAME = "Umbral A (ex.6)";
  inline std::string WINDOW_A_SPEC_NAME = "Spectrum A (ex.6)";
  inline std::string WINDOW_B_NAME = "Umbral B (ex.6)";
  inline std::string WINDOW_B_SPEC_NAME = "Spectrum B (ex.6)";

  inline std::string MODE = "Option [0-6]";
  inline std::string FILTER = "Filter Strength [0-100]";

  inline char WIN_KEY = 'd';

  int MAX_STRENGH = 100;
  float PI = 3.14159265;
  float UMBRAL_CASE_4 = 0.6;
  float UMBRAL_CASE_5 = 0.4;
  
  enum FILTER_PARAMS {
    VERTICAL,
    HORIZONTAL,
    INSIDE,
    OUTSIDE
  };

  typedef enum _filterMode {
    LOW_PASS_FILTER = 0,
    HIGH_PASS_FILTER
  } filterMode;

}

// -------- Inline Functions ----------------------------------------------

inline float rgb2hue(const float r, const float g, const float b){
  return std::acos(
    ( ((r-g) + (r-b))/2 )/
    ( std::sqrt((std::pow(r-g,2) + (r-b)*(g-b))) )
  );

}

// -------- Self-Made Functions ----------------------------------------------
namespace CVFunctions {

std::vector<cv::Mat> multichannelDFT(const cv::Mat &image_rgb){
  // [DEPRECATED] Used to compute DFT channel by chhanel.

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
// [DEPRECATED] Used to combine multiple DFT comples channels 
// into an RGB Spectrum image
{
    cv::Mat result;
    std::vector<cv::Mat> spectrums;

    for (int i = 0; i < (int)complex_channels.size(); ++i) {
        spectrums.push_back(spectrum(complex_channels[i]));
    }

    cv::merge(spectrums, result);
    return result;
}

cv::Mat bgr2hsi(const cv::Mat & rgb_image)
// Used to convert an image from BGR channels to HSI
{
  double r,g,b;
  double H,S,I;
  cv::Mat hsi(rgb_image.rows, rgb_image.cols, rgb_image.type());

  for (int i = 0; i < rgb_image.rows; i++) {
    for (int j = 0; j < rgb_image.cols; j++) {
      b = (double) rgb_image.at<cv::Vec3b>(i,j)[0];
      g = (double) rgb_image.at<cv::Vec3b>(i,j)[1];
      r = (double) rgb_image.at<cv::Vec3b>(i,j)[2];

      H = rgb2hue(r,g,b);
      S = 1 - ((3.0 * std::min(r, std::min(b,g)) / (r+g+b)));
      I = (r+g+b) / 3.0;

      if (b > g) H = 2*CVParams::PI - H;

      hsi.at<cv::Vec3b>(i, j)[0] = ((H * 180)/ CVParams::PI) ;
      // hsi.at<cv::Vec3b>(i, j)[0] = 255*((H * 180)/ PI)/360 ;
      hsi.at<cv::Vec3b>(i, j)[1] = S*255;
      hsi.at<cv::Vec3b>(i, j)[2] = I;
    }
  }

  return hsi;
}

cv::Mat substract_channels(const cv::Mat & mat1, const cv::Mat & mat2) {
  // Used to make H1 - H2 where [H1,H2] are multiple channeled images
  // ¿Directly H1 - H2 not allowed by specifications?

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
  // [DEPRECATED] Generates a rectangle and remove the intersection between the imagen and the filter
  // Reason: Not practical in exercise 6, improved with build_filter()
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

cv::Mat createHorizFilter(const cv::Mat &image, const CVParams::filterMode mode,
                          const int slider_val)
// [DEPRECATED] Builds only a rectangle based filter matrix.
// Improved with build_filter()
{
  float inside = (float) 0;  // Values inside the box
  float outside = (float) 0; // Values outside the box

  if (mode == CVParams::LOW_PASS_FILTER) inside = 1;
  else if (mode == CVParams::HIGH_PASS_FILTER) outside = 1;

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

cv::Mat build_filter(const cv::Mat &image, float strength, CVParams::FILTER_PARAMS direction, CVParams::FILTER_PARAMS mode) {
    // Builds a filter matrix given the following parameters:
    // -> Image: For exact dimension specifications
    // -> Strength: Percentage of the image to be filtered
    // -> Direction: Building orientation -> VERTICAL/HORIZONTAL
    // -> Mode: Reversed building: Inside/Outside Filtering
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

// -------- Window Management Functions ----------------------------------------------
void initWindow()
// Create window at the beggining
{
  if (CVParams::running) return;
  CVParams::running = true;

  // Show images in a different windows
  cv::namedWindow(CVParams::WINDOW_NAME);
  // create Trackbar and add to a window
  cv::createTrackbar(CVParams::MODE, CVParams::WINDOW_NAME, nullptr, 6, 0); 
  cv::createTrackbar(CVParams::FILTER, CVParams::WINDOW_NAME, nullptr, 100, 0); 
}

void hideDebug() {
  cv::destroyWindow(CVParams::WINDOW_A_NAME);
  cv::destroyWindow(CVParams::WINDOW_B_NAME);
  cv::destroyWindow(CVParams::WINDOW_A_SPEC_NAME);
  cv::destroyWindow(CVParams::WINDOW_B_SPEC_NAME);
}

}

// -------- Main Function ----------------------------------------------

namespace computer_vision
{

CVGroup CVSubscriber::processing(
  const cv::Mat in_image_rgb,
  const cv::Mat in_image_depth,
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
const
{

  // Declarations
  cv::Mat out_image_rgb, out_image_depth, preprocessed_image, hsv, hsi, complex_image, filtered_image, gray_image;
  cv::Mat image_ab, filter_a, filter_b, filter, filtered_complex_image, complex_a, complex_b;
  std::vector<cv::Mat> dft_channels;
  CVParams::FILTER_PARAMS mode, direction;
  int strength;
  int mode_param;

  // Initiailizing
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
  CVFunctions::initWindow();

  // Obtaining Parameter
  strength = cv::getTrackbarPos(CVParams::FILTER, CVParams::WINDOW_NAME);
  mode_param = cv::getTrackbarPos(CVParams::MODE, CVParams::WINDOW_NAME);
    
  switch (mode_param)
  {
  
  // Convert Image to HSI 
  case 1:
    preprocessed_image = CVFunctions::bgr2hsi(in_image_rgb);
    break;

  // Convert Image to HSI and show HSV-HSI 
  case 2:
    cv::cvtColor(in_image_rgb, hsv, cv::COLOR_BGR2HSV);
    hsi = CVFunctions::bgr2hsi(in_image_rgb);
    preprocessed_image = CVFunctions::substract_channels(hsv, hsi);
    break;

  // Show Spectrum
  case 3:
    cv::cvtColor(in_image_rgb, preprocessed_image, cv::COLOR_RGB2GRAY);
    complex_image = computeDFT(preprocessed_image);
    preprocessed_image = spectrum(complex_image);
    break;


  // Case 4: Maintain Horizontal Frecuencies
  // Case 5: Remove Horizontal Frecuencies
  // Code designed to be reused
  case 4:
  case 5:

    // Selecting Filtering Matrix Parameters
    mode = mode_param == 4 ? CVParams::OUTSIDE : CVParams::INSIDE;
    direction = CVParams::HORIZONTAL;

    // Changing to GrayScale
    cv::cvtColor(in_image_rgb, gray_image, cv::COLOR_RGB2GRAY);

    // Filtering Spectrum 
    complex_image = fftShift(computeDFT(gray_image));
    filter = CVFunctions::build_filter(complex_image, (float)strength/CVParams::MAX_STRENGH, direction, mode);
    cv::mulSpectrums(complex_image, filter, filtered_complex_image, 0);
    complex_image = fftShift(filtered_complex_image);
    cv::idft(complex_image, filtered_image, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    // Normalizing
    cv::normalize(filtered_image, preprocessed_image, 0, 1, cv::NORM_MINMAX);

    break;

  // Case 4 + Case 5 + Umbralizing + OR
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

  // Show Windows
  bool key_pressed = false;
  if (mode_param == 6 && cv::waitKey(5) == CVParams::WIN_KEY) {
    key_pressed = true;
  }

  // If change of mode OR closing in mode 6 -> close
  if ((CVParams::extra_active && mode_param != 6) || (key_pressed && CVParams::extra_active)) {
    CVFunctions::hideDebug();
    CVParams::extra_active = false;
    key_pressed = false;
  }

  // If pressed and not not active -> Open
  if (key_pressed && !CVParams::extra_active) {
    cv::imshow(CVParams::WINDOW_A_NAME, image_a);
    cv::imshow(CVParams::WINDOW_B_NAME, image_b);
    cv::imshow(CVParams::WINDOW_A_SPEC_NAME, spectrum_a);
    cv::imshow(CVParams::WINDOW_B_SPEC_NAME, spectrum_b);
    CVParams::extra_active = true;
  }

  cv::imshow(CVParams::WINDOW_NAME, preprocessed_image);
  cv::waitKey(1);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision
