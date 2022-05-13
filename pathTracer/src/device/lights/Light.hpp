#ifndef LIGHT_HPP
#define LIGHT_HPP
#pragma once

#include "../Sampling.hpp"

// LI
// evaluate light emitted radiance

// PDF
// probability with respect to the solid angle

// SAMPLE
// Importance sampling light is form interest especially if the light source covers
// a very small section of the hemisphere. This fact would introduce unwanted noise to
// the image. A much better approach would be to prefer samples which are inside the
// area of the hemisphere where the light is visible.
// 
// To make the process more efficient, one can chose a bidirectional method. However,
// this is not used here.

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#pragma region "AREA LIGHT"

Float3 powerAreaLight(InterfaceStruct const& i, LightStruct const& light)
{
	//return light.emit * triangleArea(i.TRI[0], i.TRI[1], i.TRI[2]) * PI;
	return { 0.0f };
}

Float3 liAreaLight(InterfaceStruct const& i, LightStruct const& light, Float3 const& L)
{
	//return dot(i.Ng, L) > 0.0f ? light.emit : Float3{ 0.0f };
	return { 0.0f };
}

Float pdfAreaLight(InterfaceStruct const& i)
{
	//return 1.0f / triangleArea(i.TRI[0], i.TRI[1], i.TRI[2]);
	return { 0.0f };
}

void sampleAreaLight(InterfaceStruct const& i, LightStruct const& light);

#pragma endregion




#endif // !LIGHT_HPP
