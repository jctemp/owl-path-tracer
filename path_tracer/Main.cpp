
#include "utils/parser.hpp"
#include "application.hpp"

#include "owl.hpp"

#include <fmt/core.h>

#include <filesystem>

using namespace owl;

int main(int argc, char **argv)
{
    program_data pdata{};
    test_data tdata{};
    init_program_data(pdata, tdata, std::filesystem::current_path().string() + "/assets");

    owl_data data{};
    init_owl_data(data);

    bind_sbt_data(pdata, data);

    if (tdata.vec_values.empty())
        test_loop(data, pdata, tdata, tdata.flt_values);
    else
        test_loop(data, pdata, tdata, tdata.vec_values);

    destroy_context(data.owl_context);
}
