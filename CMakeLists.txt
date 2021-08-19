cmake_minimum_required(VERSION 3.14)

project(siamFCppwarp LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(CMAKE_BUILD_TYPE MATCHES "release")
   set(Torch_DIR   D:/environment/libtorch-win-shared-with-deps-1.9.0+cpu/libtorch/)

 ELSEif(CMAKE_BUILD_TYPE MATCHES "Debug")
      set(Torch_DIR  D:/environment/libtorch-win-shared-with-deps-debug-1.9.0+cpu/libtorch/)
        endif()

message(STATUS "Torch_Dir : ${Torch_Dir}")
message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")

find_package(Torch REQUIRED)
find_package(Opencv REQUIRED)

include_directories(

    siamfcpp
    )

#  Test
# tool
#"Test/*cpp"
#"tool/*cpp"
file(GLOB_RECURSE sourcefile

"siamfcpp/*cpp"
    )

set(PROJECT_SOURCES

        main.cpp
        qml.qrc
        ${sourcefile}
)
add_executable(siamFCppwarp
    ${PROJECT_SOURCES})



target_link_libraries(siamFCppwarp

  PRIVATE ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  )
