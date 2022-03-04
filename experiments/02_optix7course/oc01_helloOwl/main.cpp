#include <simpleLogger.hpp>
#include <owl/owl.h>

extern "C" int main(int argc, char** argv)
{
	OWLContext context{ owlContextCreate(nullptr, 1) };
	owlContextDestroy(context);
	OK("Successfully created and destroyed OWL context object");
}