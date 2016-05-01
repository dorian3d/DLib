cmake_minimum_required(VERSION 2.6)
project(DLib)

option(BUILD_DUtils   "Build DUtils (basic c++ functions)." ON)
option(BUILD_DUtilsCV "Build DUtilsCV (OpenCV functions, requires DUtils)." ON)
option(BUILD_DVision  "Build DVision (computer vision functions, requires DUtilsCV)." ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
endif()

if(BUILD_DUtils)
  set(HDRS include/DUtils/
    include/DUtils/BinaryFile.h      include/DUtils/DUtils.h          include/DUtils/FileModes.h       include/DUtils/Math.hpp          
    include/DUtils/STL.h             include/DUtils/Timestamp.h
    include/DUtils/ConfigFile.h      include/DUtils/DebugFunctions.h  include/DUtils/LUT.h             include/DUtils/Profiler.h        
    include/DUtils/StringFunctions.h include/DUtils/dirent_win.h
    include/DUtils/DException.h      include/DUtils/FileFunctions.h   include/DUtils/LineFile.h        include/DUtils/Random.h          
    include/DUtils/TimeManager.h)
  set(SRCS 
    src/DUtils/BinaryFile.cpp      src/DUtils/DebugFunctions.cpp  src/DUtils/LUT.cpp             src/DUtils/Profiler.cpp        
    src/DUtils/StringFunctions.cpp src/DUtils/Timestamp.cpp
    src/DUtils/ConfigFile.cpp      src/DUtils/FileFunctions.cpp   src/DUtils/LineFile.cpp        src/DUtils/Random.cpp          
    src/DUtils/TimeManager.cpp)
endif(BUILD_DUtils)

if(BUILD_DUtilsCV)  
  set(HDRS 
    include/DUtilsCV/CvVersion.h       include/DUtilsCV/Drawing.h         include/DUtilsCV/Geometry.h        include/DUtilsCV/Mat.h             
    include/DUtilsCV/Types.h
    include/DUtilsCV/DUtilsCV.h        include/DUtilsCV/GUI.h             include/DUtilsCV/IO.h              include/DUtilsCV/Transformations.h
    ${HRDS})
  set(SRCS 
    src/DUtilsCV/Drawing.cpp         src/DUtilsCV/Geometry.cpp        src/DUtilsCV/Mat.cpp             src/DUtilsCV/Types.cpp
    src/DUtilsCV/GUI.cpp             src/DUtilsCV/IO.cpp              src/DUtilsCV/Transformations.cpp
    ${SRCS})
endif(BUILD_DUtilsCV)

if(BUILD_DVision)
  set(HDRS
    include/DVision/DVision.h        include/DVision/HSolver.h        include/DVision/Matches.h
    include/DVision/PMVSCamera.h     include/DVision/PixelPointFile.h
    include/DVision/BundleCamera.h   include/DVision/FSolver.h        include/DVision/ImageFunctions.h include/DVision/PLYFile.h        
    include/DVision/PatchFile.h      include/DVision/SurfSet.h
    ${HDRS})
  set(SRCS
    src/DVision/FSolver.cpp        src/DVision/ImageFunctions.cpp src/DVision/PLYFile.cpp
    src/DVision/PatchFile.cpp      src/DVision/SurfSet.cpp
    src/DVision/BundleCamera.cpp   src/DVision/HSolver.cpp        src/DVision/Matches.cpp        src/DVision/PMVSCamera.cpp     
    src/DVision/PixelPointFile.cpp
    ${SRCS})

  find_package(Boost QUIET)  # For dynamic_bitset
  if (Boost_FOUND)
    set(HDRS include/DVision/BRIEF.h ${HDRS})
    set(SRCS src/DVision/BRIEF.cpp ${SRCS})
  endif(Boost_FOUND)
endif(BUILD_DVision)


if(BUILD_DUtilsCV OR BUILD_DVision)
  find_package(OpenCV REQUIRED)
endif(BUILD_DUtilsCV OR BUILD_DVision)

set(LIB_SHARED "SHARED")
if(WIN32)
  set(LIB_SHARED "STATIC")
endif(WIN32)

include_directories(include/DUtils include/DUtilsCV include/DVision ${OpenCV_INCLUDE_DIRS})
add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

configure_file(src/DLib.cmake.in
  "${PROJECT_BINARY_DIR}/DLibConfig.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_DUtils)
  install(DIRECTORY include/DUtils DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
if(BUILD_DUtilsCV)
  install(DIRECTORY include/DUtilsCV DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
if(BUILD_DVision)
  install(DIRECTORY include/DVision DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DLibConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DLibConfig.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DLib/)
