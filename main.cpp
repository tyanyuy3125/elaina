#include <iostream>
#include "exec.h"

#include "core/common.h"
#include "core/device/context.h"

using namespace elaina;

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <conf_path>" << std::endl;
        return 1;
    }
    gpContext = Context::SharedPtr(new Context());
    fs::path conf_path(argv[1]);
    run_expr(conf_path);
    return 0;
}
