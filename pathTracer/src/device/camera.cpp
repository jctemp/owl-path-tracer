#include "camera.hpp"

camera_data to_camera_data(camera const& c, ivec2 const& buffer_size)
{
	float const aspect{static_cast<float>(buffer_size.x) / static_cast<float>(buffer_size.y)};

	float const theta{ c.vertical_fov * PI / 180.0f};
	float const h{ tanf(theta / 2) };
	float const viewport_height{ 2.0f * h };
	float const viewport_width{ aspect * viewport_height };

	owl::vec3f const origin{ c.look_from };
	owl::vec3f const w{ normalize(c.look_from - c.look_at) };
	owl::vec3f const u{ normalize(cross(c.look_up, w)) };
	owl::vec3f const v{ normalize(cross(w, u)) };

	owl::vec3f const horizontal{ viewport_width * u };
	owl::vec3f const vertical{ viewport_height * v };
	owl::vec3f const llc{ origin - horizontal / 2.0f - vertical / 2.0f - w };

	return camera_data{ origin, llc, horizontal, vertical };
}
