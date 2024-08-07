#ifndef MYSLAM_BACKEND_INVERSE_DEPTH_H
#define MYSLAM_BACKEND_INVERSE_DEPTH_H

#include "backend/vertex.h"

namespace myslam {
namespace backend {

/**
 * 以逆深度形式存储的顶点
 */
class VertexInverseDepth : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexInverseDepth() : Vertex(1) {}

    virtual std::string TypeInfo() const { return "VertexInverseDepth"; }//父类是纯虚类，需要在子类中实现此方法
};

}
}

#endif
