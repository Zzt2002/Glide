# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.4.1)

get_filename_component(SHADERC_SRC
        ${ANDROID_NDK}/sources/third_party/shaderc
        ABSOLUTE)

# Import shaderc lib in NDK
add_library(shaderc_lib STATIC IMPORTED)
set_target_properties(shaderc_lib PROPERTIES IMPORTED_LOCATION
        ${SHADERC_SRC}/libs/${ANDROID_STL}/${ANDROID_ABI}/libshaderc.a)

# Build application's shared lib
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -Werror -D VK_USE_PLATFORM_ANDROID_KHR")

# Force export ANativeActivity_onCreate(),
# Refer to: https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

add_library(native_app_glue STATIC
        ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
add_library(glide SHARED
        android_main.cpp
        graphics.cpp
        sensors.cpp
        ${CMAKE_SOURCE_DIR}/vulkan_wrapper/vulkan_wrapper.cpp)
target_include_directories(glide PRIVATE
        ${GLMINC_PREFIX}
        ${ANDROID_NDK}/sources/android/native_app_glue
        ${CMAKE_SOURCE_DIR}/vulkan_wrapper
        ${SHADERC_SRC}/third_party/spirv-tools/include
        ${SHADERC_SRC}/third_party/spirv-tools/include/spirv-tools
        ${ANDROID_NDK}/sources/third_party/shaderc/include)

find_library(log-lib log)
target_link_libraries(glide android log shaderc_lib native_app_glue)
