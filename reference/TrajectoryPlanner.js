/**
 * 轨迹规划器
 * 使用三次样条插值生成平滑轨迹
 */

/**
 * CubicSpline helper class for 1D interpolation.
 * Implements a natural cubic spline.
 */
class CubicSpline {
  constructor(x, y) {
    if (x.length !== y.length) {
      throw new Error("X and Y arrays must have the same length.");
    }
    if (x.length < 2) {
      throw new Error("Need at least 2 points for interpolation.");
    }

    this.x = x;
    this.y = y;
    this.length = x.length;
    this.secondDerivatives = this.computeSecondDerivatives();
  }

  computeSecondDerivatives() {
    const n = this.length;
    const m = new Array(n).fill(0).map(() => new Array(n).fill(0));
    const d = new Array(n).fill(0);
    const y2 = new Array(n).fill(0);

    // Setup tridiagonal system
    for (let i = 1; i < n - 1; i++) {
      const sig = (this.x[i] - this.x[i - 1]) / (this.x[i + 1] - this.x[i - 1]);
      const p = sig * y2[i - 1] + 2;
      y2[i] = (sig - 1) / p;
      d[i] =
        (6 *
          ((this.y[i + 1] - this.y[i]) / (this.x[i + 1] - this.x[i]) -
            (this.y[i] - this.y[i - 1]) / (this.x[i] - this.x[i - 1]))) /
        (this.x[i + 1] - this.x[i - 1]);
      d[i] = (d[i] - sig * d[i - 1]) / p;
    }

    // Back substitution for natural spline (y'' = 0 at ends)
    y2[n - 1] = 0;
    y2[0] = 0;
    for (let k = n - 2; k >= 1; k--) {
      y2[k] = y2[k] * y2[k + 1] + d[k];
    }

    return y2;
  }

  interpolate(t) {
    if (t < this.x[0] || t > this.x[this.length - 1]) {
      // Clamp to ends
      if (t < this.x[0]) return this.y[0];
      return this.y[this.length - 1];
    }

    // Binary search to find the correct interval
    let low = 0;
    let high = this.length - 1;
    let k = 0;
    while (high - low > 1) {
      k = Math.floor((low + high) / 2);
      if (this.x[k] > t) {
        high = k;
      } else {
        low = k;
      }
    }
    k = low; // k is the index of the interval start

    const h = this.x[k + 1] - this.x[k];
    if (h === 0) {
      return this.y[k];
    }

    const a = (this.x[k + 1] - t) / h;
    const b = (t - this.x[k]) / h;

    const result =
      a * this.y[k] +
      b * this.y[k + 1] +
      (((a * a * a - a) * this.secondDerivatives[k] +
        (b * b * b - b) * this.secondDerivatives[k + 1]) *
        (h * h)) /
        6.0;

    return result;
  }
}

class TrajectoryPlanner {
  /**
   * Plan a trajectory using cubic spline interpolation over waypoints.
   * @param {Array<Object>} waypoints - Array of waypoint objects. Each object is a map of jointName -> jointValue.
   * @param {number} resolution - The number of points to generate for the final trajectory.
   * @returns {Array<Object>|null} The interpolated trajectory or null if planning is not possible.
   */
  plan(waypoints, resolution = 100) {
    if (!waypoints || waypoints.length < 2) {
      console.warn("Cannot plan trajectory with less than 2 waypoints.");
      return null;
    }

    // 过滤关节名称，排除非关节属性，Filter joint names, exclude non-joint properties
    const allKeys = Object.keys(waypoints[0]);
    const jointNames = allKeys.filter(
      (key) => !["endEffectorPose"].includes(key)
    );
    const numWaypoints = waypoints.length;

    // Use waypoint index as the independent variable 'x'
    const x = Array.from({ length: numWaypoints }, (_, i) => i);

    const jointSplines = {};

    try {
      // For each joint, create a spline
      for (const jointName of jointNames) {
        const y = waypoints.map((waypoint) => waypoint[jointName]);
        jointSplines[jointName] = new CubicSpline(x, y);
      }
    } catch (error) {
      console.error("Error creating splines:", error);
      return null;
    }

    // Generate the dense trajectory
    const trajectory = [];
    for (let i = 0; i <= resolution; i++) {
      const t = (numWaypoints - 1) * (i / resolution);
      const point = {};
      for (const jointName of jointNames) {
        point[jointName] = jointSplines[jointName].interpolate(t);
      }
      trajectory.push(point);
    }

    return trajectory;
  }

  /**
   * Plan a trajectory using Linear Segments with Parabolic Blends (LSPB).
   * This creates a trapezoidal velocity profile for each segment.
   * @param {Array<Object>} waypoints - Array of waypoint objects.
   * @param {number} totalDuration - The total time for the entire trajectory in seconds.
   * @param {number} blendTimeRatio - The ratio of segment time used for acceleration/deceleration (0 to 0.5).
   * @param {number} resolutionPerSegment - The number of points to generate PER SEGMENT.
   * @returns {Array<Object>|null} The interpolated trajectory or null if planning is not possible.
   */
  planLSPB(
    waypoints,
    totalDuration = 10,
    blendTimeRatio = 0.2,
    resolutionPerSegment = 50
  ) {
    if (!waypoints || waypoints.length < 2) {
      console.warn("Cannot plan LSPB trajectory with less than 2 waypoints.");
      return null;
    }

    // 过滤关节名称，排除非关节属性，Filter joint names, exclude non-joint properties
    const allKeys = Object.keys(waypoints[0]);
    const jointNames = allKeys.filter(
      (key) => !["endEffectorPose"].includes(key)
    );
    const finalTrajectory = [];

    // Calculate the total "distance" in joint space to distribute duration proportionally
    let totalJointSpaceDistance = 0;
    const segmentDistances = [];
    for (let i = 0; i < waypoints.length - 1; i++) {
      let segmentDistance = 0;
      for (const jointName of jointNames) {
        // 只计算关节角度的距离，不包括夹爪，Only calculate joint angles distance, exclude gripper
        if (jointName !== "gripper") {
          segmentDistance += Math.pow(
            waypoints[i + 1][jointName] - waypoints[i][jointName],
            2
          );
        }
      }
      segmentDistance = Math.sqrt(segmentDistance);
      segmentDistances.push(segmentDistance);
      totalJointSpaceDistance += segmentDistance;
    }
    if (totalJointSpaceDistance < 1e-6) totalJointSpaceDistance = 1; // Avoid division by zero

    // Iterate through each segment (between two waypoints)
    for (let i = 0; i < waypoints.length - 1; i++) {
      const startWaypoint = waypoints[i];
      const endWaypoint = waypoints[i + 1];

      // Assign duration to this segment proportionally
      const segmentDuration =
        totalDuration * (segmentDistances[i] / totalJointSpaceDistance);
      const blendTime = segmentDuration * blendTimeRatio;

      // Generate points for the current segment
      for (let t_step = 0; t_step < resolutionPerSegment; t_step++) {
        const t = (t_step / resolutionPerSegment) * segmentDuration;
        const trajectoryPoint = {};

        // For each joint, calculate the interpolated value
        for (const jointName of jointNames) {
          const q0 = startWaypoint[jointName];
          const q1 = endWaypoint[jointName];
          const deltaQ = q1 - q0;

          // Handle zero blend time to avoid division by zero
          if (blendTime <= 1e-6 || 2 * blendTime >= segmentDuration) {
            // If blend time is negligible or too large, fallback to linear interpolation
            trajectoryPoint[jointName] = q0 + deltaQ * (t / segmentDuration);
            continue;
          }

          // 对夹爪使用线性插值，对关节使用LSPB，Use linear interpolation for gripper, LSPB for joints
          if (jointName === "gripper") {
            trajectoryPoint[jointName] = q0 + deltaQ * (t / segmentDuration);
          } else {
            // Corrected acceleration calculation for joints
            const acceleration =
              deltaQ / (blendTime * (segmentDuration - blendTime));

            let q_t;
            if (t >= 0 && t < blendTime) {
              // Acceleration phase
              q_t = q0 + 0.5 * acceleration * t * t;
            } else if (t >= blendTime && t <= segmentDuration - blendTime) {
              // Constant velocity phase
              const v = acceleration * blendTime;
              q_t =
                q0 +
                0.5 * acceleration * blendTime * blendTime +
                v * (t - blendTime);
            } else {
              // t > segmentDuration - blendTime && t <= segmentDuration
              // Deceleration phase
              const t_decel = t - (segmentDuration - blendTime);
              const v_blend = acceleration * blendTime;
              q_t =
                q1 -
                0.5 *
                  acceleration *
                  (blendTime - t_decel) *
                  (blendTime - t_decel);
            }
            trajectoryPoint[jointName] = q_t;
          }
        }
        finalTrajectory.push(trajectoryPoint);
      }
    }

    // Add the very last waypoint to ensure the trajectory ends perfectly
    const lastWaypoint = {};
    for (const jointName of jointNames) {
      lastWaypoint[jointName] = waypoints[waypoints.length - 1][jointName];
    }
    finalTrajectory.push(lastWaypoint);

    return finalTrajectory;
  }
}
