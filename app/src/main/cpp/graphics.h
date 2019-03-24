#ifndef GLIDE_GRAPHICS_H
#define GLIDE_GRAPHICS_H

typedef struct _display {
    int width, height;
} display;
void update_mvp(void *, uint32_t);
void graphics_init(android_app *);
void graphics_render();
void graphics_destroy();
void graphics_resize(android_app *);

#endif //GLIDE_GRAPHICS_H
