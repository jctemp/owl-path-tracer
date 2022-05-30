
#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include "owl.hpp"
#include "device/device_global.hpp"
#include <vector>

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

void init_data(owl_data& data);

void init_world(owl_data& data, std::vector<geom>& geoms);

#endif APPLICATION_HPP
