//
// Description: 
// Created by sundong on 2021/5/10.
//

#include "nolinear.hpp"

using namespace lm_for_homography;

int main(){
    std::cout << "----------Optimizer Homography Matrix----------" << std::endl;
    //原始四点坐标
    cv::Point2f srcPoint1(2.41018, 10.1169);
    cv::Point2f srcPoint2(2.90101, 27.1454);
    cv::Point2f srcPoint3(-4.21943, 18.423);
    cv::Point2f srcPoint4(0.83026, 13.5746);

    cv::Point2f srcPoint5(-0.840958, 6.9493);
    cv::Point2f srcPoint6(2.02222, 8.35886);
    cv::Point2f srcPoint7(-0.370828, 12.4945);
    cv::Point2f srcPoint8(1.33462, 17.7499);
    //目标四点坐标
    cv::Point2f dstPoint1(543.0f, 222.0f);
    cv::Point2f dstPoint2(396.0f, 200.0f);
    cv::Point2f dstPoint3(83.0f, 204.0f);
    cv::Point2f dstPoint4(233.0f, 208.0f);

    cv::Point2f dstPoint5(193.0f, 232.0f);
    cv::Point2f dstPoint6(529.0f, 220.0f);
    cv::Point2f dstPoint7(271.0f, 206.0f);
    cv::Point2f dstPoint8(362.0f, 202.0f);
    std::vector<cv::Point2f> src;
    src.push_back(srcPoint1);
    src.push_back(srcPoint2);
    src.push_back(srcPoint3);
    src.push_back(srcPoint4);
    src.push_back(srcPoint5);
    src.push_back(srcPoint6);
    src.push_back(srcPoint7);
    src.push_back(srcPoint8);

    std::vector<cv::Point2f> dst;
    dst.push_back(dstPoint1);
    dst.push_back(dstPoint2);
    dst.push_back(dstPoint3);
    dst.push_back(dstPoint4);
    dst.push_back(dstPoint5);
    dst.push_back(dstPoint6);
    dst.push_back(dstPoint7);
    dst.push_back(dstPoint8);
    std::vector<std::vector<cv::Point2f>> points;

    points.push_back(src);
    points.push_back(dst);
    NoLinear noliear_optimizer(points);
    E_Mat33 H_result;
    H_result = noliear_optimizer.OptimizeHomography();
    std::cout << "Init matrix:" << noliear_optimizer.GetInitHomography() << std::endl;
    std::cout << "Estimated matrix:\n" << H_result << "\n";

    return 0;
}