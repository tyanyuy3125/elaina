#include <catch2/catch_session.hpp>
#include "core/device/context.h"

using namespace elaina;

int main(int argc, char* argv[]) {
    gpContext = Context::SharedPtr(new Context());

    return Catch::Session().run(argc, argv);
}