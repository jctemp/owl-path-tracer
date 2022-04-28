#ifndef RENDERER_HPP
#define RENDERER_HPP

void init(void);
void release(void);
void setEnvironmentTexture(ImageRgb const& texture);
void add(Mesh* m);
void render(Camera const& cam, std::vector<MaterialStruct> const& materials);


#endif // !RENDERER_HPP


