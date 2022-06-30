#ifndef __DENSITY_SAMPLER_H__
#define __DENSITY_SAMPLER_H__

#include <glm/glm.hpp>
#include <vector>

typedef float* Func(glm::vec3 const&);

class DensitySampler
{
public:
    float sample(glm::vec3 const&) const;
    float add(glm::vec3 const&, std::vector<Func> const&) const;
    DensitySampler() = default;
};

#endif //DENSITY_SAMPLER_H