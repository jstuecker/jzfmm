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
        Node<dim,tvec> nodeA,
        Node<dim,tvec> nodeB,
        Params<tvec> params
    ) {
        // Opening Criterion
        // 2**levels is the per dimension extent
        // We use a scaling strategy to avoid floating point overflows in the squares
        const Vec<dim,int32_t> levelsA = lvl_vec<dim>(nodeA.level);
        const Vec<dim,int32_t> levelsB = lvl_vec<dim>(nodeB.level);
        const Vec<dim,tvec> dx = nodeA.center - nodeB.center;

        int scale_exp = max(levelsA[dim - 1], levelsB[dim - 1]);
        const tvec dx_max = absmax(dx);
        if(dx_max != tvec(0))
            scale_exp = max(scale_exp, normal_ilogb(dx_max));

        const Vec<dim,tvec> dx_scaled = mulpow2(dx, -scale_exp);
        const Vec<dim,tvec> extent_scaled =
            nonpositive_exp2<tvec>(levelsA - scale_exp)
            + nonpositive_exp2<tvec>(levelsB - scale_exp);

        // These cells have no finite multipole expansion. Use a predicate,
        // rather than an early return, to avoid another traversal branch.
        const bool unbounded = unbounded_node<dim,tvec>(nodeA.level)
            | unbounded_node<dim,tvec>(nodeB.level);
        return unbounded | (tvec(0.25) * extent_scaled.norm2()
            >= params.theta * params.theta * dx_scaled.norm2());
    }
};

#endif // OPENING_H
