
#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include "utils/mesh_loader.hpp"
#include "utils/image_buffer.hpp"
#include "owl.hpp"
#include "device/device_global.hpp"

#include <vector>
#include <string>
#include <tuple>
#include <memory>

struct entity
{
    mesh* mesh_ptr{};
    int32_t materialId{ -1 };
};

struct owl_data
{
    OWLContext owl_context;
    OWLModule owl_module;

    geom_type triangle_geom;

    group world;

    ray_gen_program ray_gen_prog;
    miss_program miss_prog;
    miss_program miss_shadow_prog;

    launch_params lp;
};

struct program_data
{
    std::string scene;
    std::string test_name;

    ivec2 buffer_size;
    int32_t max_samples;
    int32_t max_path_depth;

    bool environment_use;
    bool environment_auto;
    vec3 environment_color;
    float environment_intensity;
    image_buffer environment_map;

    camera_data camera;
    
    std::vector<std::tuple<std::string, material_data>> materials;
    std::vector<std::tuple<std::string, std::shared_ptr<mesh>>> meshes;
    std::vector<entity> entities;

    std::vector<geom> geoms{};
    std::vector<buffer> indices_buffer_list{};
    std::vector<buffer> vertices_buffer_list{};
    std::vector<buffer> normals_buffer_list{};

    buffer framebuffer;

    buffer material_buffer;
    buffer vertices_buffer;
    buffer indices_buffer;
    buffer normals_buffer;
};

void init_owl_data(owl_data& data);

void init_owl_world(owl_data& data, std::vector<geom>& geoms);

void init_program_data(program_data& pdata, test_data& tdata, std::string const& assets_path);

void bind_sbt_data(program_data& pdata, owl_data& data);

void modify_sbt(owl_data &odata, program_data &pdata, std::vector<std::tuple<std::string, material_data>> &materials,
                test_data const &test, float value);

void modify_sbt(owl_data &odata, program_data &pdata, std::vector<std::tuple<std::string, material_data>> &materials,
                test_data const &test, vec3 value);

void render_frame(owl_data& data, program_data& pdata, test_data& tdata, std::string const& values);

template<typename T>
void test_loop(owl_data& data, program_data& pdata, test_data& tdata, std::vector<T> const& values)
{
    auto vstart(values[0]);
    auto vend(values[1]);
    auto vstep{static_cast<int32_t>(tdata.step_size * 100)};
    for (int32_t i{0}; i <= 100; i += vstep)
    {
        auto c{i / 100.0f};
        auto value{vstart + (vend - vstart) * c};
        modify_sbt(data, pdata, pdata.materials, tdata, value);

        if constexpr(std::is_same_v<T, vec3>)
            render_frame(data, pdata, tdata, fmt::format("{:.1f}", fmt::join(std::vector<float>{
                    value.x, value.y, value.z}, ",")));
        else if constexpr(std::is_same_v<T, float>)
            render_frame(data, pdata, tdata, fmt::format("{:.1f}", value));
    }

}

#endif //APPLICATION_HPP
