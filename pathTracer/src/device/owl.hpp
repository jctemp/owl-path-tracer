#pragma once

#include <owl/owl.h>
#include <types.hpp>

using Context = OWLContext;
using Module = OWLModule;
using Raygen = OWLRayGen;
using VarDecl = OWLVarDecl

Context create_context(int32_t* device_ids, int32_t num_devices)
{
	return owlContextCreate(device_ids, num_devices);
}

Module create_module(Context ctx, char ptx[])
{
	return owlModuleCreate(ctx, ptx);
}

Raygen create_raygen(Context ctx, Module mod, char program_name[],
	uint64_t sizeof_var_struct, )
