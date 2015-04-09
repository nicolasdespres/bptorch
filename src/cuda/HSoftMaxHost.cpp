/**
 * Copyright (c) 2014, LIP6
 * author: B. Piwowarski
 */

#include <THC.h>
#include <fblualib/LuaUtils.h>
#include "thpp/Storage.h"
#include "thpp/Tensor.h"

namespace bptorch {

void launchUpdateOutputWithTargetKernel(
  const float* input,
  const float* class_weight,
  const float* class_bias,
  const float* mapping,
  const float* n_class_in_cluster,
  const float* class_start_indices,
  const float* target,
  const long* input_strides,
  const long* class_weight_strides,
  const long* class_score_strides,
  const long* cluster_score_strides,
  long input_size,
  long minibatch_size,
  long n_max_class_per_cluster,
  long n_clusters,
  float* class_score,
  float* class_logsum,
  float* cluster_score,
  float* cluster_logsum,
  float* output);

namespace {
    using namespace fblualib;
    using namespace thpp;   


    template<class T>
    Tensor<T> get_nested(lua_State *L, const char * name) {
        lua_getfield(L, 1, "updates");
        return luaGetFieldIfTensorChecked<T>(L, -1, name);
    }
    
    template<class T>
    struct Updates {
        Tensor<long> offsets;
        Tensor<long> nodes;
        Tensor<T> values;
        
        Updates(lua_State* L) :
            offsets(get_nested<long>(L, "offsets")),
            nodes(get_nested<long>(L, "nodes")),
            values(get_nested<T>(L, "values"))
        {}
    };


inline THCudaTensor* getFieldCudaTensor(lua_State* L, int arg,
                                        const char* name) {
  return static_cast<THCudaTensor*>(luaT_getfieldcheckudata(
                                      L, arg, name, "torch.CudaTensor"));
}
inline THCudaTensor* getCudaTensor(lua_State* L, int arg) {
  return static_cast<THCudaTensor*>(luaT_checkudata(L, arg,
                                                    "torch.CudaTensor"));
}


// --- Lua functions

int updateOutputWithTarget(lua_State* L) { 
  auto weight   = getFieldCudaTensor(L, 1, "weight");
  auto bias     = getFieldCudaTensor(L, 1, "bias");
  
  auto parents   = getFieldCudaTensor(L, 1, "parents");
  auto depth     = getFieldCudaTensor(L, 1, "depth");
  auto nleaves   = getFieldCudaTensor(L, 1, "nleaves");

  auto input    = getCudaTensor(L, 2);
  auto target   = getCudaTensor(L, 3);
  auto output   = getCudaTensor(L, 4);

  auto batch_size = THCudaTensor_size(NULL, input, 1);
  if (THCudaTensor_nDimension(NULL, input) == 1) {
      batch_size = 1;
  }

  Updates<float> updates(L);
  updates.offsets.resize({ batch_size + 1 });

  // Get CUDA pointers
  auto weight_data = THCudaTensor_data(NULL, weight);
  auto weight_strides = weight->stride;

  // HSoftMax_launchUpdateOutputWithTarget();
  return 0;
}

int updateGradInput(lua_State* L) {
  
  return 0;
}

int accGradParameters(lua_State* L) {
 

  return 0;
}


const luaL_Reg functions[] = {
  {"HSoftMax_updateOutputWithTarget", updateOutputWithTarget},
  // {"HSoftMax_updateGradInput", updateGradInput},
  // {"HSoftMax_accGradParameters", accGradParameters},
  {nullptr, nullptr},
};

} // namespace

void initHSMCuda(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, functions, "nn");
  lua_pop(L, 1);
}


} // bptorch
