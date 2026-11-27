cmake_minimum_required(VERSION 3.16)
project(relocate_orb_baseline LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.4 REQUIRED)
find_package(Pangolin REQUIRED)

set(ORB_SLAM3_ROOT "" CACHE PATH "Built ORB-SLAM3 source directory")
if(NOT EXISTS "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so")
    message(FATAL_ERROR "Set ORB_SLAM3_ROOT to the pinned ORB-SLAM3 checkout built in Release")
endif()
add_executable(orb_fixed_map orb_fixed_map.cc)
target_compile_features(orb_fixed_map PRIVATE cxx_std_17)
# The upstream Release build uses this flag. Eigen types cross the shared-library boundary.
target_compile_options(orb_fixed_map PRIVATE -march=native)
target_include_directories(orb_fixed_map SYSTEM PRIVATE
    "${ORB_SLAM3_ROOT}" "${ORB_SLAM3_ROOT}/include" "${ORB_SLAM3_ROOT}/include/CameraModels"
    "${ORB_SLAM3_ROOT}/Thirdparty/Sophus" ${Pangolin_INCLUDE_DIRS})
target_link_libraries(orb_fixed_map PRIVATE
    Eigen3::Eigen ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
    "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so"
    "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so")
