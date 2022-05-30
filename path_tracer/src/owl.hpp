
#ifndef PATH_TRACER_OWL_HPP
#define PATH_TRACER_OWL_HPP

#include <owl/owl.h>
#include "types.hpp"

using context = OWLContext;
using module = OWLModule;
using ray_gen_program = OWLRayGen;
using miss_program = OWLMissProg;
using launch_params = OWLParams;
using var_decl = OWLVarDecl[];
using geom_type = OWLGeomType;
using geom_kind = OWLGeomKind;
using geom = OWLGeom;
using texture = OWLTexture;
using buffer = OWLBuffer;
using owl_type = OWLDataType;
using group = OWLGroup;

inline context create_context(int32_t* device_ids, int32_t const num_devices)
{
    auto const ctx = owlContextCreate(device_ids, num_devices);
    owlContextSetRayTypeCount(ctx, 2);
    return ctx;
}

inline void destroy_context(context ctx)
{
    owlContextDestroy(ctx);
}

inline module create_module(context const ctx, char ptx[])
{
    return owlModuleCreate(ctx, ptx);
}

inline ray_gen_program create_ray_gen_program(context const ctx, module const mod, char const program_name[],
                                              uint64_t const sizeof_var_struct, var_decl vars)
{
    return owlRayGenCreate(ctx, mod, program_name, sizeof_var_struct, vars, -1);
}

inline miss_program create_miss_program(context const ctx, module const mod, char const program_name[],
                                        uint64_t const sizeof_var_struct, var_decl vars)
{
    return owlMissProgCreate(ctx, mod, program_name, sizeof_var_struct, vars, -1);
}

inline launch_params create_launch_params(context const ctx, uint64_t const sizeof_var_struct, var_decl vars)
{
    return owlParamsCreate(ctx, sizeof_var_struct, vars, -1);
}

inline geom_type create_geom_type(context const ctx, geom_kind const kind, uint64_t const sizeof_var_struct,
                                  var_decl vars)
{
    return owlGeomTypeCreate(ctx, kind, sizeof_var_struct, vars, -1);
}

inline void geom_type_closest_hit_program(geom_type const type, module const mod, char const program_name[],
                                          int32_t const ray_type = 0)
{
    owlGeomTypeSetClosestHit(type, ray_type, mod, program_name);
}

inline geom create_geom(context const ctx, geom_type const type)
{
    return owlGeomCreate(ctx, type);
}

inline buffer create_device_buffer(context const ctx, owl_type const type, uint64_t const count, void const* data)
{
    return owlDeviceBufferCreate(ctx, type, count, data);
}

inline buffer create_pinned_host_buffer(context const ctx, owl_type const type, uint64_t const size)
{
    return owlHostPinnedBufferCreate(ctx, type, size);
}

inline void set_triangle_vertices(geom g, buffer b, uint64_t size, uint64_t sizeof_type)
{
    owlTrianglesSetVertices(g, b, size, sizeof_type, 0);
}

inline void set_triangle_indices(geom g, buffer b, uint64_t size, uint64_t sizeof_type)
{
    owlTrianglesSetIndices(g, b, size, sizeof_type, 0);
}

inline group create_triangle_geom_group(context const ctx, uint64_t const size, geom* geoms)
{
    return owlTrianglesGeomGroupCreate(ctx, size, geoms);
}

inline group create_instance_group(context const ctx, uint64_t const size, group* geoms)
{
    return owlInstanceGroupCreate(ctx, size, geoms);
}

inline void build_group_acceleration_structure(group g)
{
    owlGroupBuildAccel(g);
}

inline uint32_t const* buffer_to_pointer(buffer b, int32_t device_id)
{
    return reinterpret_cast<uint32_t const*>(owlBufferGetPointer(b, device_id));
}

template<typename V>
void set_field(ray_gen_program destination, char const identifier[], V value)
{
    if constexpr (std::is_same_v<V, buffer>)
        owlRayGenSetBuffer(destination, identifier, value);
    else if constexpr (std::is_same_v<V, group>)
        owlRayGenSetGroup(destination, identifier, value);
    else if constexpr (std::is_same_v<V, texture>)
        owlRayGenSetTexture(destination, identifier, value);
    else if constexpr (std::is_same_v<V, bool>)
        owlRayGenSet1b(destination, identifier, value);
    else if constexpr (std::is_same_v<V, int32_t>)
        owlRayGenSet1i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, vec2>)
        owlRayGenSet2f(destination, identifier, (owl2f const&) value);
    else if constexpr (std::is_same_v<V, vec3>)
        owlRayGenSet3f(destination, identifier, (owl3f const&) value);
    else if constexpr (std::is_same_v<V, ivec2>)
        owlRayGenSet2i(destination, identifier, (owl2i const&) value);
    else if constexpr (std::is_same_v<V, ivec3>)
        owlRayGenSet3i(destination, identifier, (owl3i const&) value);
    else if constexpr (std::is_same_v<V, uvec2>)
        owlRayGenSet2ui(destination, identifier, (owl2ui const&) value);
    else if constexpr (std::is_same_v<V, uvec3>)
        owlRayGenSet3ui(destination, identifier, (owl3ui const&) value);
    else if constexpr (std::is_same_v<V, void*>)
        owlRayGenSetRaw(destination, identifier, value);
    else
        throw std::runtime_error("unsupported type V");
}

template<typename V>
void set_field(miss_program destination, char const identifier[], V value)
{
    if constexpr (std::is_same_v<V, buffer>)
        owlMissProgSetBuffer(destination, identifier, value);
    else if constexpr (std::is_same_v<V, group>)
        owlMissProgSetGroup(destination, identifier, value);
    else if constexpr (std::is_same_v<V, texture>)
        owlMissProgSetTexture(destination, identifier, value);
    else if constexpr (std::is_same_v<V, bool>)
        owlMissProgSet1b(destination, identifier, value);
    else if constexpr (std::is_same_v<V, int32_t>)
        owlMissProgSet1i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, vec2>)
        owlMissProgSet2f(destination, identifier, value);
    else if constexpr (std::is_same_v<V, vec3>)
        owlMissProgSet3f(destination, identifier, value);
    else if constexpr (std::is_same_v<V, ivec2>)
        owlMissProgSet2i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, ivec3>)
        owlMissProgSet3i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, uvec2>)
        owlMissProgSet2ui(destination, identifier, value);
    else if constexpr (std::is_same_v<V, uvec3>)
        owlMissProgSet3ui(destination, identifier, value);
    else if constexpr (std::is_same_v<V, void*>)
        owlMissProgSetRaw(destination, identifier, value);
    else
        throw std::runtime_error("unsupported type V");
}

template<typename V>
void set_field(launch_params destination, char const identifier[], V value)
{
    if constexpr (std::is_same_v<V, buffer>)
        owlParamsSetBuffer(destination, identifier, value);
    else if constexpr (std::is_same_v<V, group>)
        owlParamsSetGroup(destination, identifier, value);
    else if constexpr (std::is_same_v<V, texture>)
        owlParamsSetTexture(destination, identifier, value);
    else if constexpr (std::is_same_v<V, bool>)
        owlParamsSet1b(destination, identifier, value);
    else if constexpr (std::is_same_v<V, float>)
        owlParamsSet1f(destination, identifier, value);
    else if constexpr (std::is_same_v<V, int32_t>)
        owlParamsSet1i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, vec2>)
        owlParamsSet2f(destination, identifier, value);
    else if constexpr (std::is_same_v<V, vec3>)
        owlParamsSet3f(destination, identifier, (owl3f const&) value);
    else if constexpr (std::is_same_v<V, ivec2>)
        owlParamsSet2i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, ivec3>)
        owlParamsSet3i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, uvec2>)
        owlParamsSet2ui(destination, identifier, value);
    else if constexpr (std::is_same_v<V, uvec3>)
        owlParamsSet3ui(destination, identifier, value);
    else if constexpr (std::is_same_v<V, void*>)
        owlParamsSetRaw(destination, identifier, value);
    else
    {
        throw std::runtime_error("unsupported type V");
    }
}

template<typename V>
void set_field(geom destination, char const identifier[], V value)
{
    if constexpr (std::is_same_v<V, buffer>)
        owlGeomSetBuffer(destination, identifier, value);
    else if constexpr (std::is_same_v<V, group>)
        owlGeomSetGroup(destination, identifier, value);
    else if constexpr (std::is_same_v<V, texture>)
        owlGeomSetTexture(destination, identifier, value);
    else if constexpr (std::is_same_v<V, bool>)
        owlGeomSet1b(destination, identifier, value);
    else if constexpr (std::is_same_v<V, int32_t>)
        owlGeomSet1i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, vec2>)
        owlGeomSet2f(destination, identifier, value);
    else if constexpr (std::is_same_v<V, vec3>)
        owlGeomSet3f(destination, identifier, value);
    else if constexpr (std::is_same_v<V, ivec2>)
        owlGeomSet2i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, ivec3>)
        owlGeomSet3i(destination, identifier, value);
    else if constexpr (std::is_same_v<V, uvec2>)
        owlGeomSet2ui(destination, identifier, value);
    else if constexpr (std::is_same_v<V, uvec3>)
        owlGeomSet3ui(destination, identifier, value);
    else if constexpr (std::is_same_v<V, void*>)
        owlGeomSetRaw(destination, identifier, value);
    else
        throw std::runtime_error("unsupported type V");
}

inline void build_optix(context const ctx)
{
    owlBuildPrograms(ctx);
    owlBuildPipeline(ctx);
    owlBuildSBT(ctx);
}

inline texture create_texture(context const ctx, ivec2 const dims, uint32_t const* dest)
{
    return owlTexture2DCreate(
            ctx,
            OWL_TEXEL_FORMAT_RGBA8,
            dims.x, dims.y,
            dest,
            OWL_TEXTURE_NEAREST,
            OWL_TEXTURE_CLAMP);
}

inline void destroy_texture(texture tex)
{
    owlTexture2DDestroy(tex);
}

#endif //PATH_TRACER_OWL_HPP