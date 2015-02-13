#include "luaT.h"
#include "THC.h"


LUA_EXTERNC DLL_EXPORT int luaopen_libbptorch(lua_State *L);

int luaopen_libbptorch(lua_State *L)
{
  lua_newtable(L);

  return 1;
}
