
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
    std::string assets_path{ std::filesystem::current_path().string() + "/assets" };
    init_program_data(pdata, tdata, assets_path);

    owl_data data{};
    init_owl_data(data);

    bind_sbt_data(pdata, data, assets_path);

    if (tdata.vec_values.empty())
        test_loop(data, pdata, tdata, tdata.flt_values);
    else
        test_loop(data, pdata, tdata, tdata.vec_values);

    destroy_context(data.owl_context);
}
