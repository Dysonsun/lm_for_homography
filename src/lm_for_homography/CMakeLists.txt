cmake_minimum_required(VERSION 2.8.3)
project(lm_for_homography)

SET(CMAKE_BUILD_TYPE "DEBUG")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_compile_options(-std=c++14)
# add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
        message_generation
        std_srvs
        )



set(ALL_TARGET_LIBRARIES "")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories("/usr/local/include/eigen3")
include_directories(${Eigen3_INCLUDE_DIRS})


list(APPEND ALL_TARGET_LIBRARIES ${OpenCV_LIBRARIES} ${CERES_LIBRARIES} ${Eigen3_LIBRARIES})

include_directories(include ${catkin_INCLUDE_DIRS})
catkin_package()


file(GLOB_RECURSE ALL_SRCS "src/*.cpp")
file(GLOB_RECURSE NODE_SRCS "src/*_node.cpp")
list(REMOVE_ITEM ALL_SRCS ${NODE_SRCS})

add_executable(main_node src/main_node.cpp ${ALL_SRCS})
target_link_libraries(main_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

install(TARGETS
        main_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
