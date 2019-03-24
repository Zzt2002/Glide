#include <cstring>
#include <android/log.h>
#include <android/looper.h>
#include <android/sensor.h>

const char *kS_TAG = "Sensors";
#define LOGI(...) ((void) __android_log_print(ANDROID_LOG_INFO, kS_TAG, __VA_ARGS__))
#define LOGE(...) ((void) __android_log_print(ANDROID_LOG_ERROR, kS_TAG, __VA_ARGS__))
#define GET_DEFAULT(type) ASensorManager_getDefaultSensor(_ps->manager, type);
typedef struct _context {
    float rotation_quat[4];
    ASensorManager *manager;
    /* ASensorRef accelerometer; */
    ASensorRef rotation_vector;
    ASensorEventQueue *r_queue;
} context;
extern const char *package_name;
context _s, *_ps = &_s;
const int looper_id = 1;

void sensors_init() {
    _ps->manager = ASensorManager_getInstanceForPackage(package_name);
    /* _ps->accelerometer = GET_DEFAULT(ASENSOR_TYPE_ACCELEROMETER); */
    _ps->rotation_vector = GET_DEFAULT(ASENSOR_TYPE_GAME_ROTATION_VECTOR);
    _ps->r_queue = ASensorManager_createEventQueue(_ps->manager, ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS), looper_id,
                                                   nullptr, nullptr);
    ASensorEventQueue_enableSensor(_ps->r_queue, _ps->rotation_vector);
}

float *rotation_quat_pointer() {
    return _ps->rotation_quat;
}

void sensors_retrieve() {
    if(ALooper_pollAll(-1, nullptr, nullptr, nullptr) == looper_id) {
        ASensorEvent data;
        if(ASensorEventQueue_getEvents(_ps->r_queue, &data, 1)) {
            memcpy(_ps->rotation_quat, data.vector.v, 3 * sizeof(float));
        }
    }
}

void sensors_destroy() {
    ASensorEventQueue_disableSensor(_ps->r_queue, _ps->rotation_vector);
    ASensorManager_destroyEventQueue(_ps->manager, _ps->r_queue);
}
