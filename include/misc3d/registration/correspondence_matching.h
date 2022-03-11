#pragma once

#include <vector>

#include <open3d/pipelines/registration/Feature.h>

namespace misc3d {

namespace registration {

// the choise of matching algorithm depends on the feature dimension and size
// if features have large dimension and large size, the annoy method is better
// otherwise, the flann method is better
enum class MatchMethod {
    FLANN = 0,
    ANNOY = 1,
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
     * @return std::pair<std::vector<size_t>, std::vector<size_t>>
     */
    virtual std::pair<std::vector<size_t>, std::vector<size_t>> Match(
        const open3d::pipelines::registration::Feature& src,
        const open3d::pipelines::registration::Feature& dst) const = 0;

    /**
     * @brief Match two corresponding points with there feature matrix, where
     * rows is feature dimension and cols is feature size.
     *
     * @param src
     * @param dst
     * @return std::pair<std::vector<size_t>, std::vector<size_t>>
     */
    virtual std::pair<std::vector<size_t>, std::vector<size_t>> Match(
        const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst) const = 0;

    /**
     * @brief Get the Matcher Type
     *
     * @return MatcherType
     */
    MatchMethod GetMatcherType() const { return match_method_; }

protected:
    CorrespondenceMatcher(MatchMethod type) : match_method_(type) {}

protected:
    MatchMethod match_method_;
};

/**
 * @brief ANN based correspondences matching using Flann or Annoy as backend
 *
 */
class ANNMatcher : public CorrespondenceMatcher {
public:
    /**
     * @brief Construct a ANNMatcher object
     *
     */
    ANNMatcher() : CorrespondenceMatcher(MatchMethod::FLANN), n_tress_(0) {}

    ANNMatcher(const MatchMethod& method)
        : CorrespondenceMatcher(method), n_tress_(4) {}

    ANNMatcher(const MatchMethod& method, int n_trees)
        : CorrespondenceMatcher(method), n_tress_(n_trees) {}

    std::pair<std::vector<size_t>, std::vector<size_t>> Match(
        const open3d::pipelines::registration::Feature& src,
        const open3d::pipelines::registration::Feature& dst) const override;

    std::pair<std::vector<size_t>, std::vector<size_t>> Match(
        const Eigen::MatrixXd& src,
        const Eigen::MatrixXd& dst) const override;

private:
    int n_tress_;
};

}  // namespace registration

}  // namespace misc3d
