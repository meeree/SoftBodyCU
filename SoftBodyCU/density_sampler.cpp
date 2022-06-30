#include "density_sampler.h"
#include <algorithm>

float sphere(glm::vec3 const& p, float rad) { return p.x * p.x + p.y * p.y + p.z * p.z - rad*rad; }
float f(glm::vec3 const& p) { return p.z * p.z - p.x * p.x - p.y * p.y; }
float umbrella(glm::vec3 const& p) { return p.y * p.y * p.z - p.x * p.x; }
float g(glm::vec3 const& p) { return 1.0 - (p.x * p.x + p.y * p.y + p.z * p.z); }
float gstar(glm::vec3 const& p) { return 1.0 - (pow(p.x, 50) + pow(p.y, 50) + pow(p.z, 50)); }
float h(glm::vec3 const& p) { return (2 * p.y * (p.y * p.y - 3 * p.x * p.x) - (9 * p.z * p.z - 1)) * (1 - p.z * p.z) + pow(p.x * p.x + p.y * p.y, 2); }
float crossCap(glm::vec3 const& p) { return 4 * (pow(p.x, 4) + p.x * p.x * p.y * p.y + p.x * p.x * p.z * p.z + p.x * p.x * p.z) + pow(p.y, 4) + p.y * p.y * p.z * p.z - p.y * p.y; }
float torus(glm::vec3 const& p, float R, float a) { return pow(p.x * p.x + p.y * p.y + p.z * p.z + R * R - a * a, 2) - R * R * (p.x * p.x + p.z * p.z);}

float DensitySampler::sample(glm::vec3 const& p) const
{
    return sphere(p, 2);
    return std::max<float>(-p.z, sphere(glm::vec3(p.x, p.y * 1.9 / 2.5, p.z * 1.9 / 2.5), 1.9));
    return std::max<float>(fabs(p.x), std::max<float>(fabs(p.y), fabs(p.z))) - 1.5;
    return std::min<float>(std::min<float>(sphere(p + glm::vec3(1.0), 1.0), sphere(p, 1.2)), sphere(p - glm::vec3(-1.2, 0.0, 0.0), 1.0));
}

//float DensitySampler::add(glm::vec3 const& p, std::vector<float (*)(glm::vec3 const&)> const& samplers) const
//{
//    std::vector<float> data(samplers.size());
//    std::transform(samplers.begin(), samplers.end(), data.begin(), [&](float (*sampler)(glm::vec3 const&))
//    {return sampler(p); });
//    return *std::min_element(data.begin(), data.end());
//    //    return std::accumulate(data.begin(), data.end(), 0.0f)/data.size();
//}
