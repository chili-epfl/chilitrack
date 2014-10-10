#ifndef STATS_H
#define STATS_H

struct Stats
{
    int matches;
    int inliers;
    double ratio;
    int keypoints;
    int duration; //ms

    Stats() : matches(0),
        inliers(0),
        ratio(0),
        keypoints(0),
        duration(0)
    {}

    Stats& operator+=(const Stats& op) {
        matches += op.matches;
        inliers += op.inliers;
        ratio += op.ratio;
        keypoints += op.keypoints;
        duration += op.duration;
        return *this;
    }
    Stats& operator/=(int num)
    {
        matches /= num;
        inliers /= num;
        ratio /= num;
        keypoints /= num;
        duration /= num;
        return *this;
    }
};

#endif // STATS_H
