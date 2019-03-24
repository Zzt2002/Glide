#ifndef GLIDE_SENSORS_H
#define GLIDE_SENSORS_H

void sensors_init();
void sensors_retrieve();
float *rotation_quat_pointer();
void sensors_destroy();

#endif //GLIDE_SENSORS_H
