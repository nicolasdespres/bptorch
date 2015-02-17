#include <lua.hpp>

namespace facebook { namespace deeplearning { namespace torch {
    void initHSM(lua_State* L);
}}}

namespace bptorch {
    void initHSoftMax(lua_State *L);
}

using namespace facebook::deeplearning::torch;
using namespace bptorch;

#include <iostream>

extern "C" int luaopen_libbptorch(lua_State* L) {
  initHSM(L);
  initHSoftMax(L);
  return 0;
}