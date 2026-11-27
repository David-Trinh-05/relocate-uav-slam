# Fixed-map relocalization evaluation

Build the map on a dedicated mapping run. Query it with separate runs and use the same map snapshot across methods. Keep the target map fixed during evaluation and record its identifier and hash. Split development and held-out data by query run before tuning thresholds.

## Incident definition

Record an incident ID, query run ID, map version, camera masking interval, and the timestamp of the first unmasked stereo pair. Replace both images with blank frames at their original timestamps and keep the IMU samples unchanged. Keep a clean replay for comparison. Record trials in which tracking never fails separately so they do not inflate the recovery rate. Log entries into the relocalization path as well as map resets or switches.

For each frame, log the sensor timestamp in integer nanoseconds, tracking state, map ID, whether the system published a relocalized pose, and the published IMU body pose when present. Log candidates as hypotheses until the method accepts one. The upstream offline EuRoC trajectory omits failed frames and frames in other maps, so event scoring requires live frame records.

## Scoring

Fit a single rigid transform from the target map to the ground-truth world using poses available before any incident, then freeze it. Compare body poses at the frame timestamp. Interpolate bracketed ground truth using linear position interpolation and quaternion SLERP. Precheck coverage for the observation window so incidents with missing reference poses can be reported and excluded consistently before comparing methods.

Use the first published relocalization decision starting at the first unmasked pair through five seconds after it, inclusive. The initial position threshold is 0.30 m of Euclidean error and the orientation threshold is 10° of geodesic rotation error. Apply both thresholds and freeze them before running the held-out split.

| Outcome | Rule |
| --- | --- |
| Correct first pose | Published target-map pose meets both thresholds. |
| False first pose | Published pose exceeds either threshold, or the system claims recovery with the wrong map. Count the initial error even if a later frame corrects it. |
| Timeout | No relocalization decision is published in the observation window. Map resets and new-map tracking alone do not count as recovery. |

Report all three counts with the same denominator, alongside the number of independent query runs. Report median and p95 time from the first unmasked pair to a correct first pose. Keep sensor-time recovery latency separate from replay processing time.

Compare upstream ORB-SLAM3, a single hypothesis with temporal confirmation, and multiple hypotheses with temporal confirmation. Record the actual tracker path used by upstream stereo–inertial SLAM and the first frame where the correct map region was retrieved. These records help distinguish candidate retrieval failures from candidate selection failures.
