#pragma once

#include <vector>

#include <open3d/pipelines/registration/Feature.h>

namespace misc3d {

namespace registration {

enum class MatchMethod {
    FLANN = 0,
};

/**
 * @brief Absract class for correspondences matching
 *
 */
class CorrespondenceMatcher {
public:
    virtual ~CorrespondenceMatcher() {}

    /**
     * @brief Match two corresponding points with descriptors.
     *
     * @param src
     * @param dst
     * @return common::CorrespondenceIndices
     */
    virtual std::pair<std::vector<size_t>, std::vector<size_t>> Match(
        const open3d::pipelines::registration::Feature& src,
        const open3d::pipelines::registration::Feature& dst) const = 0;

    /**
     * @brief Get the Matcher Type
     *
     * @return MatcherType
     */
    MatchMethod GetMatcherType() const { return match_metod_; }

protected:
    CorrespondenceMatcher(MatchMethod type) : match_metod_(type) {}

private:
    MatchMethod match_metod_;
};

/**
 * @brief FLANN based correspondences matching using KDTreeFlann with Open3D/OpenCV
 * backend.
 *
 */
class FLANNMatcher : public CorrespondenceMatcher {
public:
    /**
     * @brief Construct a KNNMatcher object
     *
     * @param cross_check if set to true, the cross-check is performed.
     */
    FLANNMatcher(bool cross_check = true)
        : CorrespondenceMatcher(MatchMethod::FLANN), cross_check_(cross_check) {}

    std::pair<std::vector<size_t>, std::vector<size_t>> Match(
        const open3d::pipelines::registration::Feature& src,
        const open3d::pipelines::registration::Feature& dst) const override;

private:
    bool cross_check_;
};

}  // namespace registration

}  // namespace misc3d
