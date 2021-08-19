cmake_minimum_required(VERSION 3.14)

project(siamFCppwarp LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check https://doc.qt.io/qt/deployment-android.html for more information.
# They need to be set before the find_package(...) calls below.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()
if(CMAKE_BUILD_TYPE MATCHES "release")
   set(Torch_DIR   D:/environment/libtorch-win-shared-with-deps-1.9.0+cpu/libtorch/)

 ELSEif(CMAKE_BUILD_TYPE MATCHES "Debug")
      set(Torch_DIR  D:/environment/libtorch-win-shared-with-deps-debug-1.9.0+cpu/libtorch/)
        endif()

message(STATUS "Torch_Dir : ${Torch_Dir}")
message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")
find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Quick REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Quick REQUIRED)
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

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(siamFCppwarp
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(siamFCppwarp SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(siamFCppwarp
          ${PROJECT_SOURCES}

        )
    endif()
endif()

target_compile_definitions(siamFCppwarp
  PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>)
target_link_libraries(siamFCppwarp
  PRIVATE Qt${QT_VERSION_MAJOR}::Core
  Qt${QT_VERSION_MAJOR}::Quick
  PRIVATE ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  )
