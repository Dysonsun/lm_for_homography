//
// Description: 
// Created by sundong on 2021/5/10.
//

#ifndef LM_FOR_HOMOGRAPHY_NOLINEAR_HPP
#define LM_FOR_HOMOGRAPHY_NOLINEAR_HPP

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>

//ceres优化单应性矩阵中全局变量的说明
//Eigen模板类名<类型，行，列> 实例化类名；
typedef Eigen::Matrix<double ,3,3> E_Mat33; //3*3 double类型的矩阵，用来存储单应性矩阵H
typedef Eigen::Matrix<double ,2,1> E_Mat21;  //2*1 double类型的矩阵，用来存储单个数据点
typedef Eigen::MatrixXd E_Matxx;  //多行，多列的double类型的矩阵，用来存储用到的所有点

namespace lm_for_homography{
    class NoLinear{
    public:
        NoLinear(const std::vector<std::vector<cv::Point2f>> fpts);
        E_Mat33 OptimizeHomography();
        cv::Mat GetInitHomography();

    private:
        bool InitParam();

    private:
        cv::Mat homography_H_;
        std::vector<cv::Point2f> fpts_[2];
        E_Matxx x1_;
        E_Matxx x2_;


    };

    //代价函数计算函数
    template <typename T>
    void SymmetricGeometricDistanceTerms(
            const Eigen::Matrix<T,3,3> &H, //单应性矩阵变量
            const Eigen::Matrix<T,2,1> &x1, //前一帧匹配点变量
            const Eigen::Matrix<T,2,1> &x2, //当前帧匹配点变量
            T &residual)//代价函数变量
    {
        typedef Eigen::Matrix<T,3,1> E_Mat31_T;//定义齐次坐标类型
        T forward_error[2];
        E_Mat31_T x(x1(0),x1(1),T(1.0)); //将前一帧匹配点的像素坐标转为齐次坐标
        E_Mat31_T y(x2(0),x2(1),T(1.0)); //将当前帧匹配点的像素坐标转为齐次坐标
        E_Mat31_T H_x = H*x; //计算前一帧匹配点在当前帧中的重投影坐标
        H_x/=H_x(2);  //重投影坐标归一化
        forward_error[0]=H_x(0)-y(0);  //计算重投影的x方向误差
        forward_error[1]=H_x(1)-y(1);  //计算重投影的y方向误差
        //计算重投影误差的二范数作为代价函数误差
        residual=forward_error[0]*forward_error[0]+forward_error[1]*forward_error[1];
    }

    //构建代价函数结构
    class HomographySymmetricGeometricCostFunctor{
    public:
        HomographySymmetricGeometricCostFunctor(const E_Mat21 &x,const E_Mat21 &y)
                :_x(x),_y(y){} //x,y是传入代价函数的每对匹配点
        //重定义()运算符号并求代价函数
        template <typename T>
        //homography_parameters是优化变量，此例子是单应性矩阵H,residuals是代价函数误差值都是一维向量
        bool operator()(const T*homography_parameters,T*residuals)const{
            typedef Eigen::Matrix<T,3,3> E_Mat33_T;//Eigen下的3*3double矩阵
            typedef Eigen::Matrix<T,2,1> E_Mat21_T;//Eigen下的2*1double矩阵
            E_Mat33_T cH(homography_parameters);//将data指针的一维向量转为矩阵类型．
            E_Mat21_T cx(T(_x(0)),T(_x(1)));//将opencv下的前一帧匹配点坐标转为结构中需要用的类型
            E_Mat21_T cy(T(_y(0)),T(_y(1)));//将opencv下的当前一帧匹配点坐标转为结构中需要用的类型
            SymmetricGeometricDistanceTerms<T>(cH,cx,cy,residuals[0]);//调用代价函数求解公式
            return true;
        }
        const E_Mat21 _x;
        const E_Mat21 _y;
    };
}


#endif //LM_FOR_HOMOGRAPHY_NOLINEAR_HPP
