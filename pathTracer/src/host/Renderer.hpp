#ifndef RENDERER_HPP
#define RENDERER_HPP

#include "utils/image_buffer.hpp"
#include "utils/mesh_loader.hpp"
#include <pt/Types.hpp>

void init(void);
void release(void);
void setEnvironmentTexture(image_buffer const& texture);
void add(mesh* m, entity e);
void render(Camera const& cam, std::vector<MaterialStruct> const& materials, std::vector<LightStruct> const& lights);


#endif // !RENDERER_HPP


