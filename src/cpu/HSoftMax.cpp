#include <fblualib/LuaUtils.h>
#include "thpp/Storage.h"
#include "thpp/Tensor.h"

namespace bptorch {
    
    using namespace fblualib;
    using namespace thpp;
    
    template <class T> using thOps = thpp::detail::TensorOps<T>;
    
    namespace {

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
    
    template <class T>
    int updateOutputWithTarget(lua_State* L) {
        auto weight   = luaGetFieldIfTensorChecked<T>(L, 1, "weight");
        auto bias   = luaGetFieldIfTensorChecked<T>(L, 1, "bias");
        
        auto parents   = luaGetFieldIfTensorChecked<long>(L, 1, "parents");
        auto depth     = luaGetFieldIfTensorChecked<int>(L, 1, "depth");
        auto nleaves   = luaGetFieldIfNumberChecked<long>(L, 1, "nleaves");

        auto input    = luaGetTensorChecked<T>(L, 2);
        auto target   = luaGetTensorChecked<long>(L, 3);
        auto output   = luaGetTensorChecked<T>(L, 4);
        
        auto batch_size = input.size(0);
        if (input.ndims() == 1) {
            batch_size = 1;
        }

        Updates<T> updates(L);
        updates.offsets.resize({ batch_size + 1 });
        
        try {
            // Compute the total depth
            long tdepth = 0;
            for (int i_batch = 0; i_batch < batch_size; ++i_batch) {
                T logp = 0;
                long itarget = target.at({i_batch}) - 1; // 1based->0based
                long current = parents.at({itarget}); // 1based
                tdepth += depth.at(std::abs(current) - nleaves - 1) + 1;
            }
            updates.nodes.resize({tdepth});
            updates.values.resize({tdepth});
            
            long offset = 0;
            for (int i_batch = 0; i_batch < batch_size; ++i_batch) {
                T logp = 0;
                long itarget = target.at({i_batch}) - 1; // 1based->0based
                long current = parents.at({itarget}); // 1based

                updates.offsets.at({i_batch}) = offset + 1; // 1based
                while (current != 0) {
                    T sign = 1;
                    if (current < 0) {
                        sign = -1;
                        current = -current;
                    }

                    auto ix = current - nleaves - 1;
                    if (ix < 0) {
                        throw std::runtime_error("Inconsistent tree");
                    }

                    auto current_p = 1 + std::exp(sign * (weight[ix].dot(input[i_batch]) + bias.at({ix})));
                    logp = logp - std::log(current_p);
                    
                    updates.nodes.at(offset) = current;
                    updates.values.at(offset) = sign * (1. / current_p - 1.);

                    current = parents.at({current - 1});
                    ++offset;
                }
                output.at({i_batch}) = logp;
            }
            
            updates.offsets.at({batch_size}) = offset + 1;           
        } catch(std::exception &e) {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            throw e;
        }
        // return value
        
        return 0;
    }
    
    
    
    template <class T>
    int updateGradInput(lua_State* L) {
        auto weight   = luaGetFieldIfTensorChecked<T>(L, 1, "weight");
        auto nleaves   = luaGetFieldIfNumberChecked<long>(L, 1, "nleaves");

        auto gradInput    = luaGetTensorChecked<T>(L, 2);
        auto gradOutput   = luaGetTensorChecked<T>(L, 3);
        
        Updates<T> updates(L);
        
        try {
            for(auto i = 0; i < updates.offsets.size() - 1; ++i) {
                for(auto j = updates.offsets.at(i) - 1; j < updates.offsets.at(i + 1) - 1; ++j) {
                    auto ix = updates.nodes.at(j) - nleaves - 1; // 0-based
                    gradInput[i].cadd(gradOutput.at(i) * updates.values.at(j), weight[ix]);
                }
            }
        } catch(std::exception &e) {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            throw e;
        }
        // return value
        
        return 0;
    }
    
    template <class T>
    int accGradParameters(lua_State* L) {
        auto weight      = luaGetFieldIfTensorChecked<T>(L, 1, "weight");
        auto gradWeight  = luaGetFieldIfTensorChecked<T>(L, 1, "gradWeight");
        auto gradBias    = luaGetFieldIfTensorChecked<T>(L, 1, "gradBias");
        auto nleaves     = luaGetFieldIfNumberChecked<long>(L, 1, "nleaves");
        
        auto scale       = luaGetNumberChecked<double>(L, 2);
        auto input       = luaGetTensorChecked<T>(L, 3);
        auto gradOutput  = luaGetTensorChecked<T>(L, 4);

        Updates<T> updates(L);
        
        try {
            for(auto i = 0; i < updates.offsets.size() - 1; ++i) {
                for(auto j = updates.offsets.at(i) - 1; j < updates.offsets.at(i + 1) - 1; ++j) {
                    auto ix = updates.nodes.at(j) - nleaves - 1; // 0-based
                    auto c = scale * updates.values.at(j) * gradOutput.at(i);
                    gradWeight[ix].cadd(c, input[i]);
                    gradBias.at(ix) += c;
                }
            }
        } catch(std::exception &e) {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            throw e;
        }
        // return value
        
        return 0;
    }

    
    template <class T>
    class Registerer {
    private:
        static const luaL_Reg functions_[];
    public:
        static void registerFunctions(lua_State* L);
    };
    
    template <class T>
    const luaL_Reg Registerer<T>::functions_[] = {
        {"HSoftMax_updateOutputWithTarget"        , updateOutputWithTarget<T>},
        {"HSoftMax_updateGradInput"               , updateGradInput<T>},
        {"HSoftMax_accGradParameters"             , accGradParameters<T>},
        {nullptr, nullptr},
    };
    
    template <class T>
    void Registerer<T>::registerFunctions(lua_State* L) {
        luaT_pushmetatable(L, Tensor<T>::kLuaTypeName);
        luaT_registeratname(L, functions_, "nn");
        lua_pop(L, 1);
    }
    
    
    }
    
    void initHSoftMax(lua_State* L) {
        Registerer<float>::registerFunctions(L);
        Registerer<double>::registerFunctions(L);
    }
}