#include <cassert>
#include <cstdio>
#include <vector>
#include <android/log.h>
#include <android_native_app_glue.h>
#include "shaderc/shaderc.hpp"
#include <vulkan_wrapper.h>

struct vertex {
    float x, y, z, w, r, g, b, a;
};

#define XYZ1(_x_, _y_, _z_) (_x_), (_y_), (_z_), 1.f
static const vertex vertexs[] = {
        {XYZ1(-1, -1, 1), XYZ1(1.f, 0.f, 0.f)}, {XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f)}, {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f)},
        {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 0.f)}, {XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 0.f)}, {XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 0.f)},
        // green face
        {XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 0.f)}, {XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f)}, {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
        {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)}, {XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 0.f)}, {XYZ1(1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
        // blue face
        {XYZ1(-1, 1, 1), XYZ1(0.f, 0.f, 1.f)}, {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)}, {XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f)},
        {XYZ1(-1, 1, -1), XYZ1(0.f, 0.f, 1.f)}, {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)}, {XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 1.f)},
        // yellow face
        {XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 0.f)}, {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)}, {XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f)},
        {XYZ1(1, -1, 1), XYZ1(1.f, 1.f, 0.f)}, {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)}, {XYZ1(1, -1, -1), XYZ1(1.f, 1.f, 0.f)},
        // magenta face
        {XYZ1(1, 1, 1), XYZ1(1.f, 0.f, 1.f)}, {XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f)}, {XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f)},
        {XYZ1(1, 1, -1), XYZ1(1.f, 0.f, 1.f)}, {XYZ1(-1, 1, 1), XYZ1(1.f, 0.f, 1.f)}, {XYZ1(-1, 1, -1), XYZ1(1.f, 0.f, 1.f)},
        // cyan face
        {XYZ1(1, -1, 1), XYZ1(0.f, 1.f, 1.f)}, {XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f)}, {XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f)},
        {XYZ1(-1, -1, 1), XYZ1(0.f, 1.f, 1.f)}, {XYZ1(1, -1, -1), XYZ1(0.f, 1.f, 1.f)}, {XYZ1(-1, -1, -1), XYZ1(0.f, 1.f, 1.f)},
};

const char *kG_TAG = "Graphics";
#define LOGI(...) ((void) __android_log_print(ANDROID_LOG_INFO, kG_TAG, __VA_ARGS__))
#define LOGE(...) ((void) __android_log_print(ANDROID_LOG_ERROR, kG_TAG, __VA_ARGS__))
#define VK_CALL(call) assert((call) == VK_SUCCESS)
#define GET_INSTANCE_PROC_ADDR(inst, entrypoint)                               \
    {                                                                          \
        _pg->fp##entrypoint =                                                  \
            (PFN_vk##entrypoint)vkGetInstanceProcAddr(inst, "vk" #entrypoint); \
    }
#define NUM_SAMPLES VK_SAMPLE_COUNT_1_BIT
#define NUM_DESCRIPTOR_SETS 1
#define NUM_VIEWPORTS 1
#define NUM_SCISSORS 1
#define FENCE_TIMEOUT ((int)1e9)
typedef struct {
    VkLayerProperties properties;
    std::vector<VkExtensionProperties> instance_extensions;
    std::vector<VkExtensionProperties> device_extensions;
} layer_properties;
typedef struct _swap_chain_buffers {
    VkImage image;
    VkImageView view;
} swap_chain_buffer;
typedef struct _texture_object {
    VkSampler sampler;

    VkImage image;
    VkImageLayout imageLayout;

    VkDeviceMemory mem;
    VkImageView view;
    int32_t tex_width, tex_height;
} texture_object;
typedef struct _context {
    std::vector<layer_properties> instance_layer_properties;
    PFN_vkCreateAndroidSurfaceKHR fpCreateAndroidSurfaceKHR;
    VkInstance inst;
    std::vector<VkPhysicalDevice> gpus;
    uint32_t gpu_count;

    uint32_t queue_family_count;
    VkPhysicalDeviceProperties gpu_props;
    std::vector<VkQueueFamilyProperties> queue_props;
    VkPhysicalDeviceMemoryProperties memory_properties;
    uint32_t graphics_queue_family_index;
    uint32_t present_queue_family_index;
    VkQueue graphics_queue;
    VkQueue present_queue;
    VkDevice device;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    std::vector<const char *> instance_extension_names;
    std::vector<VkExtensionProperties> instance_extension_properties;
    std::vector<const char *> device_extension_names;
    std::vector<VkExtensionProperties> device_extension_properties;
    VkSurfaceKHR surface;
    VkFramebuffer *framebuffers;
    VkFormat format;
    uint32_t swapchain_image_count;
    VkSwapchainKHR swap_chain;
    std::vector<swap_chain_buffer> buffers;
    VkSemaphore image_acquired_semaphore;
    struct {
        VkFormat format;
        VkImage image;
        VkDeviceMemory mem;
        VkImageView view;
    } depth;
    std::vector<texture_object> textures;

    void *mvp;
    uint32_t mvp_size;
    struct {
        VkBuffer buf;
        VkDeviceMemory mem;
        VkDescriptorBufferInfo buffer_info;
    } uniform_data;

    struct {
        VkDescriptorImageInfo image_info;
    } texture_data;
    VkDeviceMemory stagingMemory;
    VkImage stagingImage;

    struct {
        VkBuffer buf;
        VkDeviceMemory mem;
        VkDescriptorBufferInfo buffer_info;
    } vertex_buffer;
    VkVertexInputBindingDescription vi_binding;
    VkVertexInputAttributeDescription vi_attribs[2];

    VkPipelineLayout pipeline_layout;
    std::vector<VkDescriptorSetLayout> desc_layout;
    VkPipelineCache pipeline_cache;
    VkRenderPass render_pass;
    VkPipeline pipeline;

    VkPipelineShaderStageCreateInfo shaderStages[2];

    VkDescriptorPool desc_pool;
    std::vector<VkDescriptorSet> desc_set;

    uint32_t current_buffer;
    VkViewport viewport;
    VkRect2D scissor;
} context;
typedef struct _display {
    int width, height;
} display;
display display_0;
context _g, *_pg = &_g;

bool include_depth = true, use_texture = false;
bool memory_type_from_properties(uint32_t type_bits, VkFlags requirements_mask, uint32_t *type_index) {
    for (uint32_t i = 0; i < _pg->memory_properties.memoryTypeCount; i++) {
        if ((type_bits & 1) == 1) {
            if ((_pg->memory_properties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                *type_index = i;
                return true;
            }
        }
        type_bits >>= 1;
    }
    return false;
}

static const char *vert_shader_text =
        "#version 400\n"
        "#extension GL_ARB_separate_shader_objects : enable\n"
        "#extension GL_ARB_shading_language_420pack : enable\n"
        "layout (std140, binding = 0) uniform bufferVals {\n"
        "    mat4 mvp;\n"
        "} myBufferVals;\n"
        "layout (location = 0) in vec4 pos;\n"
        "layout (location = 1) in vec4 inColor;\n"
        "layout (location = 0) out vec4 outColor;\n"
        "void main() {\n"
        "   outColor = inColor;\n"
        "   gl_Position = myBufferVals.mvp * pos;\n"
        "}\n";

static const char *frag_shader_text =
        "#version 400\n"
        "#extension GL_ARB_separate_shader_objects : enable\n"
        "#extension GL_ARB_shading_language_420pack : enable\n"
        "layout (location = 0) in vec4 color;\n"
        "layout (location = 0) out vec4 outColor;\n"
        "void main() {\n"
        "   outColor = color;\n"
        "}\n";

struct shader_type_mapping {
    VkShaderStageFlagBits vkshader_type;
    shaderc_shader_kind shaderc_type;
};
static const shader_type_mapping shader_map_table[] = {
        {VK_SHADER_STAGE_VERTEX_BIT, shaderc_glsl_vertex_shader},
        {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, shaderc_glsl_tess_control_shader},
        {VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, shaderc_glsl_tess_evaluation_shader},
        {VK_SHADER_STAGE_GEOMETRY_BIT, shaderc_glsl_geometry_shader},
        {VK_SHADER_STAGE_FRAGMENT_BIT, shaderc_glsl_fragment_shader},
        {VK_SHADER_STAGE_COMPUTE_BIT, shaderc_glsl_compute_shader},
};
shaderc_shader_kind MapShadercType(VkShaderStageFlagBits vkShader) {
    for (auto shader : shader_map_table) {
        if (shader.vkshader_type == vkShader) {
            return shader.shaderc_type;
        }
    }
    return shaderc_glsl_infer_from_source;
}
bool GLSLtoSPV(const VkShaderStageFlagBits shader_type, const char *pshader, std::vector<unsigned int> &spirv) {
    // On Android, use shaderc instead.
    shaderc::Compiler compiler;
    shaderc::SpvCompilationResult module =
            compiler.CompileGlslToSpv(pshader, strlen(pshader), MapShadercType(shader_type), "shader");
    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOGE("Error: Id=%d, Msg=%s", module.GetCompilationStatus(), module.GetErrorMessage().c_str());
        return false;
    }
    spirv.assign(module.cbegin(), module.cend());
    return true;
}
void init_global_extension_properties(layer_properties &layer_props) {
    VkExtensionProperties *instance_extensions;
    uint32_t instance_extension_count;
    char *layer_name = nullptr;
    layer_name = layer_props.properties.layerName;

    do {
        if(vkEnumerateInstanceExtensionProperties(layer_name, &instance_extension_count, nullptr))
            return;
        if (instance_extension_count == 0)
            return;
        layer_props.instance_extensions.resize(instance_extension_count);
        instance_extensions = layer_props.instance_extensions.data();
    } while (vkEnumerateInstanceExtensionProperties(layer_name, &instance_extension_count, instance_extensions) == VK_INCOMPLETE);
}
void init_device_extension_properties(layer_properties &layer_props) {
    VkExtensionProperties *device_extensions;
    uint32_t device_extension_count;
    char *layer_name = nullptr;
    layer_name = layer_props.properties.layerName;

    do {
        if (vkEnumerateDeviceExtensionProperties(_pg->gpus[0], layer_name, &device_extension_count, nullptr)) return;
        if (device_extension_count == 0)
            return;
        layer_props.device_extensions.resize(device_extension_count);
        device_extensions = layer_props.device_extensions.data();
    } while (vkEnumerateDeviceExtensionProperties(_pg->gpus[0], layer_name, &device_extension_count, device_extensions) == VK_INCOMPLETE);
}
void init_instance() {
    assert(init_vulkan());
    LOGI("Loaded Vulkan APIs.");

    _pg->instance_extension_names.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    _pg->instance_extension_names.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
    _pg->device_extension_names.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    VkLayerProperties *vk_props = nullptr;
    uint32_t instance_layer_count;
    do {
        vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr);
        if (instance_layer_count == 0)
            break;
        vk_props = (VkLayerProperties *)realloc(vk_props, instance_layer_count * sizeof(VkLayerProperties));
    } while (vkEnumerateInstanceLayerProperties(&instance_layer_count, vk_props) == VK_INCOMPLETE);

    for (uint32_t i = 0; i < instance_layer_count; i++) {
        layer_properties layer_props;
        layer_props.properties = vk_props[i];
        init_global_extension_properties(layer_props);
        _pg->instance_layer_properties.push_back(layer_props);
    }
    free(vk_props);
    VkApplicationInfo app_info;
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = nullptr;
    app_info.pApplicationName = "";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo inst_info;
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = nullptr;
    inst_info.flags = 0;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledLayerCount = 0;
    inst_info.ppEnabledLayerNames = nullptr;
    inst_info.enabledExtensionCount = _pg->instance_extension_names.size();
    inst_info.ppEnabledExtensionNames = _pg->instance_extension_names.data();

    VK_CALL(vkCreateInstance(&inst_info, nullptr, &_pg->inst));
}
void query_device() {
    vkEnumeratePhysicalDevices(_pg->inst, &_pg->gpu_count, nullptr);
    assert(_pg->gpu_count > 0);
    _pg->gpus.resize(_pg->gpu_count);
    vkEnumeratePhysicalDevices(_pg->inst, &_pg->gpu_count, _pg->gpus.data());

    vkGetPhysicalDeviceQueueFamilyProperties(
            _pg->gpus[0], &_pg->queue_family_count, nullptr);
    assert(_pg->queue_family_count > 0);
    _pg->queue_props.resize(_pg->queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
            _pg->gpus[0], &_pg->queue_family_count, _pg->queue_props.data());
    vkGetPhysicalDeviceMemoryProperties(_pg->gpus[0], &_pg->memory_properties);
    vkGetPhysicalDeviceProperties(_pg->gpus[0], &_pg->gpu_props);
    for (auto& layer_props : _pg->instance_layer_properties)
        init_device_extension_properties(layer_props);
}
void init_surface(android_app *app) {
    assert(app->window);
    LOGI("Display [%d, %d]", display_0.width, display_0.height);
    GET_INSTANCE_PROC_ADDR(_pg->inst, CreateAndroidSurfaceKHR);
    VkAndroidSurfaceCreateInfoKHR surface_create_info;
    surface_create_info.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    surface_create_info.pNext = nullptr;
    surface_create_info.flags = 0;
    surface_create_info.window = app->window;
    VK_CALL(_pg->fpCreateAndroidSurfaceKHR(_pg->inst, &surface_create_info, nullptr, &_pg->surface));
}
void init_queue() {
    VkBool32 supported;
    for (uint32_t i = 0; i < _pg->queue_family_count; i++) {
        VK_CALL(vkGetPhysicalDeviceSurfaceSupportKHR(_pg->gpus[0], i, _pg->surface, &supported));
        if(supported) {
            _pg->present_queue_family_index = i;
            break;
        }
    }
    assert(supported);
    LOGI("Queue [G%d P%d]", _pg->graphics_queue_family_index, _pg->present_queue_family_index);
}
void init_surface_format() {
    uint32_t format_count;
    VK_CALL(vkGetPhysicalDeviceSurfaceFormatsKHR(_pg->gpus[0], _pg->surface, &format_count, nullptr));
    VkSurfaceFormatKHR *surf_formats = (VkSurfaceFormatKHR *)malloc(format_count * sizeof(VkSurfaceFormatKHR));
    VK_CALL(vkGetPhysicalDeviceSurfaceFormatsKHR(_pg->gpus[0], _pg->surface, &format_count, surf_formats));
    if (format_count == 1 && surf_formats[0].format == VK_FORMAT_UNDEFINED) {
        _pg->format = VK_FORMAT_B8G8R8A8_UNORM;
    } else {
        assert(format_count > 1);
        _pg->format = surf_formats[0].format;
    }
    free(surf_formats);
}
void create_device() {
    VkDeviceQueueCreateInfo queue_info = {};
    for(uint32_t i = 0; i < _pg->queue_family_count; ++i) {
        if(_pg->queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            _pg->graphics_queue_family_index = i;
        }
    }
    float queue_priorities[1] = {0.0};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.pNext = nullptr;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = queue_priorities;
    queue_info.queueFamilyIndex = _pg->graphics_queue_family_index;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = nullptr;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = _pg->device_extension_names.size();
    device_info.ppEnabledExtensionNames = device_info.enabledExtensionCount ? _pg->device_extension_names.data() : nullptr;
    device_info.enabledLayerCount = 0;
    device_info.ppEnabledLayerNames = nullptr;
    device_info.pEnabledFeatures = nullptr;
    VK_CALL(vkCreateDevice(_pg->gpus[0], &device_info, nullptr, &_pg->device));
}
void init_cmd_pool() {
    VkCommandPoolCreateInfo cmd_pool_info = {};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.pNext = nullptr;
    cmd_pool_info.queueFamilyIndex = _pg->graphics_queue_family_index;
    cmd_pool_info.flags = 0;
    VK_CALL(vkCreateCommandPool(_pg->device, &cmd_pool_info, nullptr, &_pg->cmd_pool));
}
void init_cmd_buffer() {
    VkCommandBufferAllocateInfo cmd_info = {};
    cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_info.pNext = nullptr;
    cmd_info.commandPool = _pg->cmd_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    VK_CALL(vkAllocateCommandBuffers(_pg->device, &cmd_info, &_pg->cmd));
}
void begin_cmd() {
    VkCommandBufferBeginInfo cmd_buf_info = {};
    cmd_buf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_buf_info.pNext = nullptr;
    cmd_buf_info.flags = 0;
    cmd_buf_info.pInheritanceInfo = nullptr;
    VK_CALL(vkBeginCommandBuffer(_pg->cmd, &cmd_buf_info));
}
void init_device_queue() {
    vkGetDeviceQueue(_pg->device, _pg->graphics_queue_family_index, 0, &_pg->graphics_queue);
    if (_pg->graphics_queue_family_index == _pg->present_queue_family_index)
        _pg->present_queue = _pg->graphics_queue;
    else
        vkGetDeviceQueue(_pg->device, _pg->present_queue_family_index, 0, &_pg->present_queue);
}
void init_swapchain() {
    VkSurfaceCapabilitiesKHR surf_capabilities;
    VK_CALL(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_pg->gpus[0], _pg->surface, &surf_capabilities));
    uint32_t present_mode_count;
    VK_CALL(vkGetPhysicalDeviceSurfacePresentModesKHR(_pg->gpus[0], _pg->surface, &present_mode_count, nullptr));
    VkPresentModeKHR *present_modes = (VkPresentModeKHR *)malloc(present_mode_count * sizeof(VkPresentModeKHR));
    VK_CALL(vkGetPhysicalDeviceSurfacePresentModesKHR(_pg->gpus[0], _pg->surface, &present_mode_count, present_modes));

    VkExtent2D swapchain_extent;
    if (surf_capabilities.currentExtent.width == 0xFFFFFFFF) {
        swapchain_extent.width = display_0.width;
        swapchain_extent.height = display_0.height;
        if (swapchain_extent.width < surf_capabilities.minImageExtent.width) {
            swapchain_extent.width = surf_capabilities.minImageExtent.width;
        } else if (swapchain_extent.width > surf_capabilities.maxImageExtent.width) {
            swapchain_extent.width = surf_capabilities.maxImageExtent.width;
        }

        if (swapchain_extent.height < surf_capabilities.minImageExtent.height) {
            swapchain_extent.height = surf_capabilities.minImageExtent.height;
        } else if (swapchain_extent.height > surf_capabilities.maxImageExtent.height) {
            swapchain_extent.height = surf_capabilities.maxImageExtent.height;
        }
    } else {
        // If the surface size is defined, the swap chain size must match
        swapchain_extent = surf_capabilities.currentExtent;
    }

    VkPresentModeKHR swapchain_present_mode = VK_PRESENT_MODE_FIFO_KHR;

    VkSurfaceTransformFlagBitsKHR preTransform;
    if (surf_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        preTransform = surf_capabilities.currentTransform;
    }

    // Find a supported composite alpha mode - one of these is guaranteed to be set
    VkCompositeAlphaFlagBitsKHR composite_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    VkCompositeAlphaFlagBitsKHR composite_alpha_flags[4] = {
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (uint32_t i = 0; i < sizeof(composite_alpha_flags); i++) {
        if (surf_capabilities.supportedCompositeAlpha & composite_alpha_flags[i]) {
            composite_alpha = composite_alpha_flags[i];
            break;
        }
    }

    VkSwapchainCreateInfoKHR swapchain_ci = {};
    swapchain_ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_ci.pNext = nullptr;
    swapchain_ci.surface = _pg->surface;
    swapchain_ci.minImageCount = surf_capabilities.minImageCount;
    swapchain_ci.imageFormat = _pg->format;
    swapchain_ci.imageExtent.width = swapchain_extent.width;
    swapchain_ci.imageExtent.height = swapchain_extent.height;
    swapchain_ci.preTransform = preTransform;
    swapchain_ci.compositeAlpha = composite_alpha;
    swapchain_ci.imageArrayLayers = 1;
    swapchain_ci.presentMode = swapchain_present_mode;
    swapchain_ci.oldSwapchain = VK_NULL_HANDLE;
    swapchain_ci.clipped = false;
    swapchain_ci.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    swapchain_ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    swapchain_ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_ci.queueFamilyIndexCount = 0;
    swapchain_ci.pQueueFamilyIndices = nullptr;
    uint32_t queueFamilyIndices[2] = {(uint32_t)_pg->graphics_queue_family_index, (uint32_t)_pg->present_queue_family_index};
    if (_pg->graphics_queue_family_index != _pg->present_queue_family_index) {
        swapchain_ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_ci.queueFamilyIndexCount = 2;
        swapchain_ci.pQueueFamilyIndices = queueFamilyIndices;
    }

    VK_CALL(vkCreateSwapchainKHR(_pg->device, &swapchain_ci, nullptr, &_pg->swap_chain));
    VK_CALL(vkGetSwapchainImagesKHR(_pg->device, _pg->swap_chain, &_pg->swapchain_image_count, nullptr));
    VkImage *swapchain_images = (VkImage *)malloc(_pg->swapchain_image_count * sizeof(VkImage));
    VK_CALL(vkGetSwapchainImagesKHR(_pg->device, _pg->swap_chain, &_pg->swapchain_image_count, swapchain_images));

    for (uint32_t i = 0; i < _pg->swapchain_image_count; i++) {
        swap_chain_buffer sc_buffer;

        VkImageViewCreateInfo color_image_view = {};
        color_image_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        color_image_view.pNext = nullptr;
        color_image_view.format = _pg->format;
        color_image_view.components.r = VK_COMPONENT_SWIZZLE_R;
        color_image_view.components.g = VK_COMPONENT_SWIZZLE_G;
        color_image_view.components.b = VK_COMPONENT_SWIZZLE_B;
        color_image_view.components.a = VK_COMPONENT_SWIZZLE_A;
        color_image_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        color_image_view.subresourceRange.baseMipLevel = 0;
        color_image_view.subresourceRange.levelCount = 1;
        color_image_view.subresourceRange.baseArrayLayer = 0;
        color_image_view.subresourceRange.layerCount = 1;
        color_image_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        color_image_view.flags = 0;

        sc_buffer.image = swapchain_images[i];

        color_image_view.image = sc_buffer.image;

        VK_CALL(vkCreateImageView(_pg->device, &color_image_view, nullptr, &sc_buffer.view));
        _pg->buffers.push_back(sc_buffer);
    }
    free(swapchain_images);
    _pg->current_buffer = 0;
    free(present_modes);
}
void init_depth_buffer() {
    VkImageCreateInfo image_info = {};
    _pg->depth.format = VK_FORMAT_D24_UNORM_S8_UINT;
    const VkFormat depth_format = _pg->depth.format;
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(_pg->gpus[0], depth_format, &props);
    if (props.linearTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        image_info.tiling = VK_IMAGE_TILING_LINEAR;
    } else if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    } else LOGE("No valid tiling.");

    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.pNext = nullptr;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = depth_format;
    image_info.extent.width = display_0.width;
    image_info.extent.height = display_0.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = NUM_SAMPLES;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.queueFamilyIndexCount = 0;
    image_info.pQueueFamilyIndices = nullptr;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image_info.flags = 0;

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.allocationSize = 0;
    alloc_info.memoryTypeIndex = 0;

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.pNext = nullptr;
    view_info.image = VK_NULL_HANDLE;
    view_info.format = depth_format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_R;
    view_info.components.g = VK_COMPONENT_SWIZZLE_G;
    view_info.components.b = VK_COMPONENT_SWIZZLE_B;
    view_info.components.a = VK_COMPONENT_SWIZZLE_A;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.flags = 0;

    if (depth_format == VK_FORMAT_D16_UNORM_S8_UINT || depth_format == VK_FORMAT_D24_UNORM_S8_UINT ||
        depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    VkMemoryRequirements mem_reqs;
    VK_CALL(vkCreateImage(_pg->device, &image_info, nullptr, &_pg->depth.image));
    vkGetImageMemoryRequirements(_pg->device, _pg->depth.image, &mem_reqs);

    alloc_info.allocationSize = mem_reqs.size;
    assert(memory_type_from_properties(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &alloc_info.memoryTypeIndex));

    VK_CALL(vkAllocateMemory(_pg->device, &alloc_info, nullptr, &_pg->depth.mem));
    VK_CALL(vkBindImageMemory(_pg->device, _pg->depth.image, _pg->depth.mem, 0));
    view_info.image = _pg->depth.image;
    VK_CALL(vkCreateImageView(_pg->device, &view_info, nullptr, &_pg->depth.view));
}
void update_mvp(void *mvp, uint32_t mvp_size) {
    _pg->mvp_size = mvp_size;
    if(!mvp) return;
    _pg->mvp = mvp;
    uint8_t *p_data;
    VK_CALL(vkMapMemory(_pg->device, _pg->uniform_data.mem, 0, _pg->uniform_data.buffer_info.range, 0, (void **)&p_data));
    memcpy(p_data, _pg->mvp, _pg->mvp_size);
    vkUnmapMemory(_pg->device, _pg->uniform_data.mem);
}
void init_uniform_buffer() {
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.pNext = nullptr;
    buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buf_info.size = sizeof(_pg->mvp_size);
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = nullptr;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buf_info.flags = 0;
    VK_CALL(vkCreateBuffer(_pg->device, &buf_info, nullptr, &_pg->uniform_data.buf));

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(_pg->device, _pg->uniform_data.buf, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.memoryTypeIndex = 0;

    alloc_info.allocationSize = mem_reqs.size;
    assert(memory_type_from_properties(mem_reqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &alloc_info.memoryTypeIndex));

    VK_CALL(vkAllocateMemory(_pg->device, &alloc_info, nullptr, &(_pg->uniform_data.mem)));
    VK_CALL(vkBindBufferMemory(_pg->device, _pg->uniform_data.buf, _pg->uniform_data.mem, 0));

    _pg->uniform_data.buffer_info.buffer = _pg->uniform_data.buf;
    _pg->uniform_data.buffer_info.offset = 0;
    _pg->uniform_data.buffer_info.range = sizeof(_pg->mvp_size);
}
void init_layouts(bool use_texture) {
    VkDescriptorSetLayoutBinding layout_bindings[2];
    layout_bindings[0].binding = 0;
    layout_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layout_bindings[0].descriptorCount = 1;
    layout_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    layout_bindings[0].pImmutableSamplers = nullptr;

    /* *** Texture switch */
    if (use_texture) {
        layout_bindings[1].binding = 1;
        layout_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layout_bindings[1].descriptorCount = 1;
        layout_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        layout_bindings[1].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo descriptor_layout = {};
    descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout.pNext = nullptr;
    descriptor_layout.flags = 0;
    descriptor_layout.bindingCount = use_texture ? 2 : 1;
    descriptor_layout.pBindings = layout_bindings;

    _pg->desc_layout.resize(NUM_DESCRIPTOR_SETS);
    VK_CALL(vkCreateDescriptorSetLayout(_pg->device, &descriptor_layout, nullptr, _pg->desc_layout.data()));

    VkPipelineLayoutCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_info.pNext = nullptr;
    pipeline_info.pushConstantRangeCount = 0;
    pipeline_info.pPushConstantRanges = nullptr;
    pipeline_info.setLayoutCount = NUM_DESCRIPTOR_SETS;
    pipeline_info.pSetLayouts = _pg->desc_layout.data();

    VK_CALL(vkCreatePipelineLayout(_pg->device, &pipeline_info, nullptr, &_pg->pipeline_layout));
}
void init_renderpass(bool include_depth, bool clear) {
    /* *** Renderpass */
    VkAttachmentDescription attachments[2];
    attachments[0].format = _pg->format;
    attachments[0].samples = NUM_SAMPLES;
    attachments[0].loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; /* *** */
    attachments[0].flags = 0;

    if (include_depth) {
        attachments[1].format = _pg->depth.format;
        attachments[1].samples = NUM_SAMPLES;
        attachments[1].loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments[1].flags = 0;
    }

    VkAttachmentReference color_reference = {};
    color_reference.attachment = 0;
    color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_reference = {};
    depth_reference.attachment = 1;
    depth_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.flags = 0;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = nullptr;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_reference;
    subpass.pResolveAttachments = nullptr;
    subpass.pDepthStencilAttachment = &depth_reference;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = nullptr;

    VkRenderPassCreateInfo rp_info = {};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.pNext = nullptr;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 0;
    rp_info.pDependencies = nullptr;

    VK_CALL(vkCreateRenderPass(_pg->device, &rp_info, nullptr, &_pg->render_pass));
}
void init_pipeline_cache() {
    VkPipelineCacheCreateInfo pipeline_cache;
    pipeline_cache.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pipeline_cache.pNext = nullptr;
    pipeline_cache.initialDataSize = 0;
    pipeline_cache.pInitialData = nullptr;
    pipeline_cache.flags = 0;

    VK_CALL(vkCreatePipelineCache(_pg->device, &pipeline_cache, nullptr, &_pg->pipeline_cache));
}
void load_shaders(const char *vert_shader_text, const char *frag_shader_text) {
    VkShaderModuleCreateInfo moduleCreateInfo;

    if (vert_shader_text) {
        std::vector<unsigned int> vtx_spv;
        _pg->shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        _pg->shaderStages[0].pNext = nullptr;
        _pg->shaderStages[0].pSpecializationInfo = nullptr;
        _pg->shaderStages[0].flags = 0;
        _pg->shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        _pg->shaderStages[0].pName = "main";

        assert(GLSLtoSPV(VK_SHADER_STAGE_VERTEX_BIT, vert_shader_text, vtx_spv));

        moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCreateInfo.pNext = nullptr;
        moduleCreateInfo.flags = 0;
        moduleCreateInfo.codeSize = vtx_spv.size() * sizeof(unsigned int);
        moduleCreateInfo.pCode = vtx_spv.data();
        VK_CALL(vkCreateShaderModule(_pg->device, &moduleCreateInfo, nullptr, &_pg->shaderStages[0].module));
    }

    if (frag_shader_text) {
        std::vector<unsigned int> frag_spv;
        _pg->shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        _pg->shaderStages[1].pNext = nullptr;
        _pg->shaderStages[1].pSpecializationInfo = nullptr;
        _pg->shaderStages[1].flags = 0;
        _pg->shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        _pg->shaderStages[1].pName = "main";

        assert(GLSLtoSPV(VK_SHADER_STAGE_FRAGMENT_BIT, frag_shader_text, frag_spv));

        moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCreateInfo.pNext = nullptr;
        moduleCreateInfo.flags = 0;
        moduleCreateInfo.codeSize = frag_spv.size() * sizeof(unsigned int);
        moduleCreateInfo.pCode = frag_spv.data();
        VK_CALL(vkCreateShaderModule(_pg->device, &moduleCreateInfo, nullptr, &_pg->shaderStages[1].module));
    }
}
void init_framebuffers(bool include_depth) {
    /* DEPENDS on init_depth_buffer(), init_renderpass() and
     * init_swapchain_extension() */

    VkImageView attachments[2];
    attachments[1] = _pg->depth.view;

    VkFramebufferCreateInfo fb_info = {};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.renderPass = _pg->render_pass;
    fb_info.attachmentCount = include_depth ? 2 : 1;
    fb_info.pAttachments = attachments;
    fb_info.width = display_0.width;
    fb_info.height = display_0.height;
    fb_info.layers = 1;

    uint32_t i;

    _pg->framebuffers = (VkFramebuffer *)malloc(_pg->swapchain_image_count * sizeof(VkFramebuffer));

    for (i = 0; i < _pg->swapchain_image_count; i++) {
        attachments[0] = _pg->buffers[i].view;
        VK_CALL(vkCreateFramebuffer(_pg->device, &fb_info, nullptr, &_pg->framebuffers[i]));
    }
}
void update_vertex_buffer(const void *vertex_data, uint32_t data_size) {
    uint8_t *p_data;
    VK_CALL(vkMapMemory(_pg->device, _pg->vertex_buffer.mem, 0, VK_WHOLE_SIZE , 0, (void **)&p_data));
    memcpy(p_data, vertex_data, data_size);
    vkUnmapMemory(_pg->device, _pg->vertex_buffer.mem);
}
void init_vertex_buffer(uint32_t data_size, uint32_t data_stride, bool use_texture) {
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.pNext = nullptr;
    buf_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    buf_info.size = data_size;
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = nullptr;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buf_info.flags = 0;
    VK_CALL(vkCreateBuffer(_pg->device, &buf_info, nullptr, &_pg->vertex_buffer.buf));

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(_pg->device, _pg->vertex_buffer.buf, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.memoryTypeIndex = 0;

    alloc_info.allocationSize = mem_reqs.size;
    assert(memory_type_from_properties(mem_reqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &alloc_info.memoryTypeIndex));

    VK_CALL(vkAllocateMemory(_pg->device, &alloc_info, nullptr, &(_pg->vertex_buffer.mem)));
    _pg->vertex_buffer.buffer_info.range = mem_reqs.size;
    _pg->vertex_buffer.buffer_info.offset = 0;
    VK_CALL(vkBindBufferMemory(_pg->device, _pg->vertex_buffer.buf, _pg->vertex_buffer.mem, 0));

    _pg->vi_binding.binding = 0;
    _pg->vi_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    _pg->vi_binding.stride = data_stride;

    _pg->vi_attribs[0].binding = 0;
    _pg->vi_attribs[0].location = 0;
    _pg->vi_attribs[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    _pg->vi_attribs[0].offset = 0;
    _pg->vi_attribs[1].binding = 0;
    _pg->vi_attribs[1].location = 1;
    _pg->vi_attribs[1].format = use_texture ? VK_FORMAT_R32G32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    _pg->vi_attribs[1].offset = 16;
}
void init_descriptor_pool(bool use_texture) {
    /* DEPENDS on init_uniform_buffer() and
     * init_descriptor_and_pipeline_layouts() */

    VkDescriptorPoolSize type_count[2];
    type_count[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    type_count[0].descriptorCount = 1;
    if (use_texture) {
        type_count[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        type_count[1].descriptorCount = 1;
    }

    VkDescriptorPoolCreateInfo descriptor_pool = {};
    descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool.pNext = nullptr;
    descriptor_pool.maxSets = 1;
    descriptor_pool.poolSizeCount = use_texture ? 2 : 1;
    descriptor_pool.pPoolSizes = type_count;

    VK_CALL(vkCreateDescriptorPool(_pg->device, &descriptor_pool, nullptr, &_pg->desc_pool));
}
void init_descriptor_set(bool use_texture) {
    /* DEPENDS on init_descriptor_pool() */

    VkDescriptorSetAllocateInfo alloc_info[1];
    alloc_info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info[0].pNext = nullptr;
    alloc_info[0].descriptorPool = _pg->desc_pool;
    alloc_info[0].descriptorSetCount = NUM_DESCRIPTOR_SETS;
    alloc_info[0].pSetLayouts = _pg->desc_layout.data();

    _pg->desc_set.resize(NUM_DESCRIPTOR_SETS);
    VK_CALL(vkAllocateDescriptorSets(_pg->device, alloc_info, _pg->desc_set.data()));

    VkWriteDescriptorSet writes[2];

    writes[0] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = nullptr;
    writes[0].dstSet = _pg->desc_set[0];
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &_pg->uniform_data.buffer_info;
    writes[0].dstArrayElement = 0;
    writes[0].dstBinding = 0;

    if (use_texture) {
        writes[1] = {};
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = _pg->desc_set[0];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &_pg->texture_data.image_info;
        writes[1].dstArrayElement = 0;
    }

    vkUpdateDescriptorSets(_pg->device, use_texture ? 2 : 1, writes, 0, nullptr);
}
void init_pipeline(bool include_depth, bool include_vi=true) {
    VkDynamicState dynamicStateEnables[VK_DYNAMIC_STATE_RANGE_SIZE];
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    memset(dynamicStateEnables, 0, sizeof dynamicStateEnables);
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.pNext = nullptr;
    dynamicState.pDynamicStates = dynamicStateEnables;
    dynamicState.dynamicStateCount = 0;

    VkPipelineVertexInputStateCreateInfo vi;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (include_vi) {
        vi.pNext = nullptr;
        vi.flags = 0;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &_pg->vi_binding;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = _pg->vi_attribs;
    }
    VkPipelineInputAssemblyStateCreateInfo ia;
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.pNext = nullptr;
    ia.flags = 0;
    ia.primitiveRestartEnable = VK_FALSE;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineRasterizationStateCreateInfo rs;
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.pNext = nullptr;
    rs.flags = 0;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.depthBiasEnable = VK_FALSE;
    rs.depthBiasConstantFactor = 0;
    rs.depthBiasClamp = 0;
    rs.depthBiasSlopeFactor = 0;
    rs.lineWidth = 1.0f;

    VkPipelineColorBlendStateCreateInfo cb;
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.flags = 0;
    cb.pNext = nullptr;
    VkPipelineColorBlendAttachmentState att_state[1];
    att_state[0].colorWriteMask = 0xf;
    att_state[0].blendEnable = VK_FALSE;
    att_state[0].alphaBlendOp = VK_BLEND_OP_ADD;
    att_state[0].colorBlendOp = VK_BLEND_OP_ADD;
    att_state[0].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    att_state[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    att_state[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    att_state[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cb.attachmentCount = 1;
    cb.pAttachments = att_state;
    cb.logicOpEnable = VK_FALSE;
    cb.logicOp = VK_LOGIC_OP_NO_OP;
    cb.blendConstants[0] = 1.0f;
    cb.blendConstants[1] = 1.0f;
    cb.blendConstants[2] = 1.0f;
    cb.blendConstants[3] = 1.0f;

    VkPipelineViewportStateCreateInfo vp = {};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.pNext = nullptr;
    vp.flags = 0;
    // Temporary disabling dynamic viewport on Android because some of drivers doesn't
    // support the feature.
    VkViewport viewports;
    viewports.minDepth = 0.0f;
    viewports.maxDepth = 1.0f;
    viewports.x = 0;
    viewports.y = 0;
    viewports.width = display_0.width;
    viewports.height = display_0.height;
    VkRect2D scissor;
    scissor.extent.width = display_0.width;
    scissor.extent.height = display_0.height;
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    vp.viewportCount = NUM_VIEWPORTS;
    vp.scissorCount = NUM_SCISSORS;
    vp.pScissors = &scissor;
    vp.pViewports = &viewports;
    VkPipelineDepthStencilStateCreateInfo ds;
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.pNext = nullptr;
    ds.flags = 0;
    ds.depthTestEnable = include_depth;
    ds.depthWriteEnable = include_depth;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.stencilTestEnable = VK_FALSE;
    ds.back.failOp = VK_STENCIL_OP_KEEP;
    ds.back.passOp = VK_STENCIL_OP_KEEP;
    ds.back.compareOp = VK_COMPARE_OP_ALWAYS;
    ds.back.compareMask = 0;
    ds.back.reference = 0;
    ds.back.depthFailOp = VK_STENCIL_OP_KEEP;
    ds.back.writeMask = 0;
    ds.minDepthBounds = 0;
    ds.maxDepthBounds = 0;
    ds.stencilTestEnable = VK_FALSE;
    ds.front = ds.back;

    VkPipelineMultisampleStateCreateInfo ms;
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.pNext = nullptr;
    ms.flags = 0;
    ms.pSampleMask = nullptr;
    ms.rasterizationSamples = NUM_SAMPLES;
    ms.sampleShadingEnable = VK_FALSE;
    ms.alphaToCoverageEnable = VK_FALSE;
    ms.alphaToOneEnable = VK_FALSE;
    ms.minSampleShading = 0.0;

    VkGraphicsPipelineCreateInfo pipeline;
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = nullptr;
    pipeline.layout = _pg->pipeline_layout;
    pipeline.basePipelineHandle = VK_NULL_HANDLE;
    pipeline.basePipelineIndex = 0;
    pipeline.flags = 0;
    pipeline.pVertexInputState = &vi;
    pipeline.pInputAssemblyState = &ia;
    pipeline.pRasterizationState = &rs;
    pipeline.pColorBlendState = &cb;
    pipeline.pTessellationState = nullptr;
    pipeline.pMultisampleState = &ms;
    pipeline.pDynamicState = &dynamicState;
    pipeline.pViewportState = &vp;
    pipeline.pDepthStencilState = &ds;
    pipeline.pStages = _pg->shaderStages;
    pipeline.stageCount = 2;
    pipeline.renderPass = _pg->render_pass;
    pipeline.subpass = 0;

    VK_CALL(vkCreateGraphicsPipelines(_pg->device, _pg->pipeline_cache, 1, &pipeline, nullptr, &_pg->pipeline));
}
void graphics_init(android_app *app) {
    init_instance();
    query_device();
    init_surface(app);
    init_queue();
    init_surface_format();
    create_device();
    init_cmd_pool();
    init_cmd_buffer();
    init_device_queue();
    init_swapchain();
    init_depth_buffer();
    init_uniform_buffer();
    init_layouts(use_texture);
    init_renderpass(true, true);

    /* *** Shaders */
    load_shaders(vert_shader_text, frag_shader_text);
    init_framebuffers(include_depth);
    init_vertex_buffer(sizeof(vertexs), sizeof(vertexs[0]), use_texture);
    init_descriptor_pool(use_texture);
    init_descriptor_set(use_texture);
    init_pipeline_cache();
    init_pipeline(include_depth);
}
void graphics_render() {
    update_vertex_buffer(vertexs, sizeof(vertexs));
    VkClearValue clear_values[2];
    clear_values[0].color.float32[0] = 0.2f;
    clear_values[0].color.float32[1] = 0.2f;
    clear_values[0].color.float32[2] = 0.2f;
    clear_values[0].color.float32[3] = 0.2f;
    clear_values[1].depthStencil.depth = 1.0f;
    clear_values[1].depthStencil.stencil = 0;

    VkSemaphoreCreateInfo semaphore_create_info;
    semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_create_info.pNext = nullptr;
    semaphore_create_info.flags = 0;

    VK_CALL(vkCreateSemaphore(_pg->device, &semaphore_create_info, nullptr, &_pg->image_acquired_semaphore));

    // Get the index of the next available swapchain image:
    VK_CALL(vkAcquireNextImageKHR(_pg->device, _pg->swap_chain, UINT64_MAX, _pg->image_acquired_semaphore, VK_NULL_HANDLE,
                                &_pg->current_buffer));
    // TODO: Deal with the VK_SUBOPTIMAL_KHR and VK_ERROR_OUT_OF_DATE_KHR
    // return codes

    VkRenderPassBeginInfo rp_begin;
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.pNext = nullptr;
    rp_begin.renderPass = _pg->render_pass;
    rp_begin.framebuffer = _pg->framebuffers[_pg->current_buffer];
    rp_begin.renderArea.offset.x = 0;
    rp_begin.renderArea.offset.y = 0;
    rp_begin.renderArea.extent.width = display_0.width;
    rp_begin.renderArea.extent.height = display_0.height;

    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;

    begin_cmd();
    vkCmdBeginRenderPass(_pg->cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(_pg->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pg->pipeline);
    vkCmdBindDescriptorSets(_pg->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pg->pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                            _pg->desc_set.data(), 0, nullptr);

    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(_pg->cmd, 0, 1, &_pg->vertex_buffer.buf, offsets);

    // vkCmdSetViewport(_pg->cmd, 0, NUM_VIEWPORTS, &_pg->viewport);
    // vkCmdSetScissor(_pg->cmd, 0, NUM_SCISSORS, &_pg->scissor);

    vkCmdDraw(_pg->cmd, 12 * 3, 1, 0, 0);
    vkCmdEndRenderPass(_pg->cmd);
    vkEndCommandBuffer(_pg->cmd);
    const VkCommandBuffer cmd_bufs[] = {_pg->cmd};
    VkFenceCreateInfo fence_info;
    VkFence draw_fence;
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.pNext = nullptr;
    fence_info.flags = 0;
    vkCreateFence(_pg->device, &fence_info, nullptr, &draw_fence);

    VkPipelineStageFlags pipe_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info[1] = {};
    submit_info[0].pNext = nullptr;
    submit_info[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info[0].waitSemaphoreCount = 1;
    submit_info[0].pWaitSemaphores = &_pg->image_acquired_semaphore;
    submit_info[0].pWaitDstStageMask = &pipe_stage_flags;
    submit_info[0].commandBufferCount = 1;
    submit_info[0].pCommandBuffers = cmd_bufs;
    submit_info[0].signalSemaphoreCount = 0;
    submit_info[0].pSignalSemaphores = nullptr;

    /* Queue the command buffer for execution */
    VK_CALL(vkQueueSubmit(_pg->graphics_queue, 1, submit_info, draw_fence));

    /* Now present the image in the window */
    VkPresentInfoKHR present;
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.pNext = nullptr;
    present.swapchainCount = 1;
    present.pSwapchains = &_pg->swap_chain;
    present.pImageIndices = &_pg->current_buffer;
    present.pWaitSemaphores = nullptr;
    present.waitSemaphoreCount = 0;
    present.pResults = nullptr;

    /* Make sure command buffer is finished before presenting */
    while (vkWaitForFences(_pg->device, 1, &draw_fence, VK_TRUE, FENCE_TIMEOUT) == VK_TIMEOUT);
    VK_CALL(vkQueuePresentKHR(_pg->present_queue, &present));
    vkDestroySemaphore(_pg->device, _pg->image_acquired_semaphore, NULL);
    vkDestroyFence(_pg->device, draw_fence, NULL);
}
void graphics_destroy() {
    vkDestroyPipeline(_pg->device, _pg->pipeline, nullptr);
    vkDestroyPipelineCache(_pg->device, _pg->pipeline_cache, nullptr);
    vkDestroyDescriptorPool(_pg->device, _pg->desc_pool, nullptr);
    vkDestroyBuffer(_pg->device, _pg->vertex_buffer.buf, nullptr);
    vkFreeMemory(_pg->device, _pg->vertex_buffer.mem, nullptr);
    for (uint32_t i = 0; i < _pg->swapchain_image_count; i++) {
        vkDestroyFramebuffer(_pg->device, _pg->framebuffers[i], nullptr);
    }
    free(_pg->framebuffers);
    vkDestroyShaderModule(_pg->device, _pg->shaderStages[0].module, nullptr);
    vkDestroyShaderModule(_pg->device, _pg->shaderStages[1].module, nullptr);
    vkDestroyRenderPass(_pg->device, _pg->render_pass, nullptr);
    for (uint32_t i = 0; i < NUM_DESCRIPTOR_SETS; i++)
        vkDestroyDescriptorSetLayout(_pg->device, _pg->desc_layout[i], nullptr);
    vkDestroyPipelineLayout(_pg->device, _pg->pipeline_layout, nullptr);
    for (uint32_t i = 0; i < NUM_DESCRIPTOR_SETS; i++)
        vkDestroyDescriptorSetLayout(_pg->device, _pg->desc_layout[i], nullptr);
    vkDestroyPipelineLayout(_pg->device, _pg->pipeline_layout, nullptr);
    vkDestroyBuffer(_pg->device, _pg->uniform_data.buf, nullptr);
    vkFreeMemory(_pg->device, _pg->uniform_data.mem, nullptr);
    vkDestroyImageView(_pg->device, _pg->depth.view, nullptr);
    vkDestroyImage(_pg->device, _pg->depth.image, nullptr);
    vkFreeMemory(_pg->device, _pg->depth.mem, nullptr);
    for (uint32_t i = 0; i < _pg->swapchain_image_count; i++)
        vkDestroyImageView(_pg->device, _pg->buffers[i].view, nullptr);
    vkDestroySwapchainKHR(_pg->device, _pg->swap_chain, nullptr);
    VkCommandBuffer cmd_bufs[1] = {_pg->cmd};
    vkFreeCommandBuffers(_pg->device, _pg->cmd_pool, 1, cmd_bufs);
    vkDestroyCommandPool(_pg->device, _pg->cmd_pool, nullptr);
    vkDestroyDevice(_pg->device, nullptr);
    vkDestroyInstance(_pg->inst, nullptr);
}
void graphics_resize(android_app *app) {
    vkDestroyPipeline(_pg->device, _pg->pipeline, nullptr);
    vkDestroyPipelineCache(_pg->device, _pg->pipeline_cache, nullptr);
    vkDestroyDescriptorPool(_pg->device, _pg->desc_pool, nullptr);
    vkDestroyBuffer(_pg->device, _pg->vertex_buffer.buf, nullptr);
    vkFreeMemory(_pg->device, _pg->vertex_buffer.mem, nullptr);
    for (uint32_t i = 0; i < _pg->swapchain_image_count; i++) {
        vkDestroyFramebuffer(_pg->device, _pg->framebuffers[i], nullptr);
    }
    free(_pg->framebuffers);
    vkDestroyShaderModule(_pg->device, _pg->shaderStages[0].module, nullptr);
    vkDestroyShaderModule(_pg->device, _pg->shaderStages[1].module, nullptr);
    vkDestroyRenderPass(_pg->device, _pg->render_pass, nullptr);
    for (uint32_t i = 0; i < NUM_DESCRIPTOR_SETS; i++)
        vkDestroyDescriptorSetLayout(_pg->device, _pg->desc_layout[i], nullptr);
    vkDestroyPipelineLayout(_pg->device, _pg->pipeline_layout, nullptr);
    for (uint32_t i = 0; i < NUM_DESCRIPTOR_SETS; i++)
        vkDestroyDescriptorSetLayout(_pg->device, _pg->desc_layout[i], nullptr);
    vkDestroyPipelineLayout(_pg->device, _pg->pipeline_layout, nullptr);
    vkDestroyBuffer(_pg->device, _pg->uniform_data.buf, nullptr);
    vkFreeMemory(_pg->device, _pg->uniform_data.mem, nullptr);
    vkDestroyImageView(_pg->device, _pg->depth.view, nullptr);
    vkDestroyImage(_pg->device, _pg->depth.image, nullptr);
    vkFreeMemory(_pg->device, _pg->depth.mem, nullptr);
    for (uint32_t i = 0; i < _pg->swapchain_image_count; i++)
        vkDestroyImageView(_pg->device, _pg->buffers[i].view, nullptr);
    vkDestroySwapchainKHR(_pg->device, _pg->swap_chain, nullptr);
    init_device_queue();
    init_swapchain();
    init_depth_buffer();
    init_uniform_buffer();
    init_layouts(use_texture);
    init_renderpass(true, true);
    /* *** Shaders */
    load_shaders(vert_shader_text, frag_shader_text);
    init_framebuffers(include_depth);
    init_vertex_buffer(sizeof(vertexs), sizeof(vertexs[0]), use_texture);
    init_descriptor_pool(use_texture);
    init_descriptor_set(use_texture);
    init_pipeline_cache();
    init_pipeline(include_depth);
}
