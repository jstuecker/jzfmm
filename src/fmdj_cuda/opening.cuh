#ifndef OPENING_H
#define OPENING_H

#include "common/data.cuh"
#include "common/math.cuh"

static constexpr int OPENING_BY_ANGLE = 0;

template<int opening_criterion_kind>
struct OpeningCriterion;

template<>
struct OpeningCriterion<OPENING_BY_ANGLE> {
    template<typename tvec>
    struct Params {
        tvec theta;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int dim, typename tvec>
    __device__ __forceinline__ static bool should_open(
        NodeWithExt<dim,tvec> nodeA,
        NodeWithExt<dim,tvec> nodeB,
        Params<tvec> params
    ) {
        tvec r2 = (nodeA.center - nodeB.center).norm2();
        Vec<dim,tvec> Ltot = nodeA.extent + nodeB.extent;
        tvec Lmax = Ltot[0];
        #pragma unroll
        for(int d = 1; d < dim; d++) {
            Lmax = Ltot[d] > Lmax ? Ltot[d] : Lmax;
        }
        tvec L2 = Lmax * Lmax;

        bool need_open = L2 >= params.theta * params.theta * r2;
        // also open if L2 had an overflow (and r2 is valid)
        need_open = need_open || ((isnan(L2) || isinf(L2)) && !isnan(r2));
        return need_open;
    }
};

#endif // OPENING_H
