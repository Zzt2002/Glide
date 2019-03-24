#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include "sensors.h"
#include "graphics.h"

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"

const char *kM_TAG = "Main-Module";
#define LOGI(...) ((void) __android_log_print(ANDROID_LOG_INFO, kM_TAG, __VA_ARGS__))
#define LOGE(...) ((void) __android_log_print(ANDROID_LOG_ERROR, kM_TAG, __VA_ARGS__))

const char *package_name = "glide";
extern display display_0;
android_app *app_inst;
pthread_t render_thread;
struct _transform {
    glm::mat4 projection, view, model, clip, mvp;
} transform;
void transform_apply(void) {
    transform.mvp = transform.clip * transform.projection * transform.view * transform.model;
}
void transform_init(void) {
    float fov = glm::radians(45.0f);
    if (display_0.width > display_0.height) {
        fov *= static_cast<float>(display_0.height) / static_cast<float>(display_0.width);
    }
    transform.projection = glm::perspective(fov, static_cast<float>(display_0.width) / static_cast<float>(display_0.height), 0.1f, 100.0f);
    transform.view = glm::lookAt(glm::vec3(12, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, -1, 0));
    transform.model = glm::mat4(1.0f);
    // Vulkan clip space has inverted Y and half Z.
    transform.clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f);
}
const useconds_t interval = (useconds_t)2e4;
float *rotation_quat;
void *render(void *p_arg) {
    sensors_init();
    while(!app_inst->destroyRequested) {
        sensors_retrieve();
        rotation_quat[3] = glm::sqrt(1 - (rotation_quat[0] * rotation_quat[0]
                + rotation_quat[1] * rotation_quat[1] +  rotation_quat[2] * rotation_quat[2]));
        glm::quat rotation = glm::quat(rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3]);
        transform.model = glm::mat4_cast(rotation);
        transform_apply();
        update_mvp((void *)&transform.mvp, sizeof(transform.mvp));
        graphics_render();
        usleep(interval);
    }
    sensors_destroy();
    pthread_exit(0);
}
void handle_cmd(android_app* app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            display_0.width = ANativeWindow_getWidth(app->window);
            display_0.height = ANativeWindow_getHeight(app->window);
            transform_init();
            rotation_quat = rotation_quat_pointer();
            update_mvp(nullptr, sizeof(transform.mvp));
            graphics_init(app);
            pthread_create(&render_thread, nullptr, render, nullptr);
            break;
        case APP_CMD_TERM_WINDOW:
            graphics_destroy();
            break;
        /*case 8:
            graphics_resize(app);*/
        default:
            LOGI("Event not handled: %d", cmd);
    }
}
void android_main(android_app *app) {
    app_inst = app;
    app->onAppCmd = handle_cmd;
    do {
        int events;
        android_poll_source *source;
        if (ALooper_pollAll(0, NULL, &events, (void **)&source) >= 0) {
            if (source != NULL) source->process(app, source);
        }
    } while(!app->destroyRequested);
}
