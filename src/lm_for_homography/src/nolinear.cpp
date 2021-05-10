//
// Description: 
// Created by sundong on 2021/5/10.
//

#include "nolinear.hpp"

namespace lm_for_homography{
    NoLinear::NoLinear(const std::vector<std::vector<cv::Point2f>> fpts)
    {
        assert(fpts.size() == 2);
        assert(fpts[0].size() == fpts[1].size());
        fpts_[0] = fpts[0];
        fpts_[1] = fpts[1];
        InitParam();
    }

    bool NoLinear::InitParam() {
        homography_H_ = cv::findHomography(fpts_[0],fpts_[1],cv::RANSAC,3,cv::noArray(),2000,0.995);
//        std::vector<cv::Point2f> src;
//        std::vector<cv::Point2f> dst;
//
//        for(int i = 0; i < 4; i++){
//            src.push_back(fpts_[0][i]);
//            dst.push_back(fpts_[1][i]);
//        }
//
//        homography_H_ = cv::findHomography(src, dst, CV_FM_8POINT);
        int num = fpts_[0].size();
        x1_.resize(2, num);
        x2_.resize(2, num);
        for (int i = 0; i < num; ++i) {
            x1_(0, i) = static_cast<double>(fpts_[0][i].x);
            x1_(1, i) = static_cast<double>(fpts_[0][i].y);
            x2_(0, i) = static_cast<double>(fpts_[1][i].x);
            x2_(1, i) = static_cast<double>(fpts_[1][i].y);
        }
    }

    E_Mat33 NoLinear::OptimizeHomography() {
        //数据信息判断
        assert(2==x1_.rows());
        assert(4<=x1_.cols());
        assert(x1_.rows()==x2_.rows());
        assert(x1_.cols()==x2_.cols());
        //初始化单应性矩阵H
        E_Mat33 H;
        cv::cv2eigen(homography_H_,H);//opencv中矩阵转为Eigen中矩阵
        ceres::Problem problem; //构建最小二乘问题
        //往构建的问题中添加匹配点、核函数、单应性矩阵
        for (int i = 0; i <x1_.cols() ; ++i) {
            //调用构建的代价函数结构
            HomographySymmetricGeometricCostFunctor *homography_cost_functor=
                    new HomographySymmetricGeometricCostFunctor(x1_.col(i),x2_.col(i));
            problem.AddResidualBlock(
                    //ceres自动求导求增量方程<代价函数类型，代价函数维度，优化变量维度>（代价函数）
                    new ceres::AutoDiffCostFunction<HomographySymmetricGeometricCostFunctor,1,9>
                            (homography_cost_functor),
                    new ceres::CauchyLoss(1),//添加柯西核函数
                    H.data()//添加的优化变量（矩阵转为一维向量）
            );
        }
        //ceres求解器优化配置(很多设置默认值就可以了)
        //默认最大迭代50次，误差1e-32
        ceres::Solver::Options solver_options;//实例化求解器对象
        //线性求解器的类型，用于计算Levenberg-Marquardt算法每次迭代中线性最小二乘问题的解
        solver_options.linear_solver_type=ceres::DENSE_QR;
        //记录优化过程，输出到cout位置
        solver_options.minimizer_progress_to_stdout= true;
        solver_options.max_num_iterations = 512;
        solver_options.function_tolerance = 1e-7;
        //ceres求解运算
        //实例化求解对象
        ceres::Solver::Summary summary;
        //求解
        ceres::Solve(solver_options, &problem, &summary);
        //输出优化信息
        std::cout<<"summary:\n"<<summary.BriefReport() << std::endl;
        if(!summary.IsSolutionUsable()){
            std::cerr << "Error! Solution is unusable!!!" << std::endl;
        }
        H /= H(2, 2);
        return H;
    }

    cv::Mat NoLinear::GetInitHomography() {
        return homography_H_;
    }

}