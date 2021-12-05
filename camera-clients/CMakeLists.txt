project( smartFog )

cmake_minimum_required(VERSION 3.13)

find_package( OpenCV REQUIRED )

SET(CMAKE_BUILD_TYPE Release)

# Flags de compilación de C++.
set (CMAKE_CXX_FLAGS "-O3 -g -Wall -pthread")


add_executable( camera-client
  camera-client.cpp
  camera-utils.cpp
  xml-utils/tinyxml2.cpp
  xml-utils/xml-utils.cpp)
target_link_libraries( camera-client ${OpenCV_LIBS} )
