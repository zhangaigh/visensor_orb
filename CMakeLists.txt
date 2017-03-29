cmake_minimum_required(VERSION 2.8)
project(visensor_slam)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
	set(ENABAL_PANGOLIN 1)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

	# Check C++11 or C++0x support
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	if(COMPILER_SUPPORTS_CXX11)
	   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	   add_definitions(-DCOMPILEDWITHC11)
	   message(STATUS "Using flag -std=c++11.")
	elseif(COMPILER_SUPPORTS_CXX0X)
	   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	   add_definitions(-DCOMPILEDWITHC0X)
	   message(STATUS "Using flag -std=c++0x.")
	else()
	   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
	endif()




	find_package(OpenCV REQUIRED)
	#find_package(Eigen3 3.1.0 REQUIRED)
	find_package(Boost  1.54.0		# Minimum or EXACT version e.g. 1.36.0
		 REQUIRED             		# Fail with error if Boost is not found
		 COMPONENTS serialization filesystem system	# Boost libraries by their canonical name
	)

	if (ENABAL_PANGOLIN)
		add_definitions(-DPANGONLIN_VIEWER)
		find_package(Pangolin REQUIRED)
	else (ENABAL_PANGOLIN)
	endif ()

		include_directories(${PROJECT_SOURCE_DIR}
						${PROJECT_SOURCE_DIR}/include
						${EIGEN3_INCLUDE_DIR}
						${Pangolin_INCLUDE_DIRS})
			LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/libs)

 	add_executable(visensor_orb visensor_slam.cpp)
    target_link_libraries(visensor_orb orb_slam loitorusbcam loitorimu usb-1.0 ${OpenCV_LIBS}
		${EIGEN3_LIBS}
		${Pangolin_LIBRARIES})

