/**
 * IK Solvers collection
 *
 * 提供可插拔的 IK 求解器实现。目前包含 JacobianLM6DoFIKSolver 与 DLSIKSolver。
 */

/**
 * JacobianLM6DoFIKSolver
 *
 * :param robotArm, Object - robot arm instance providing FK (updateAllJoints, getEndEffectorPosition, getEndEffectorOrientation)
 * :param jointNamesProvider, Function - returns Array<string> of joint names
 * :param jointLimitProvider, Function - returns joint limit { min, max } for a joint
 * :return: Object - solver instance with solve(...)
 */
class JacobianLM6DoFIKSolver {
  constructor(robotArm, jointNamesProvider, jointLimitProvider) {
    this.robotArm = robotArm;
    this.jointNamesProvider = jointNamesProvider;
    this.jointLimitProvider = jointLimitProvider;
    this.maxIterations = 150;
    this.positionTolerance = 0.005; // 5mm
    this.orientationTolerance = 0.05; // ~3 degrees
    this.stepSize = 0.03;
    this.debug = false; // enable per-iteration logging
  }

  /**
   * Solve 6DOF IK with redundancy (n>=6)
   *
   * :param targetPosition, THREE.Vector3 - desired position
   * :param targetOrientation, THREE.Quaternion - desired orientation
   * :param initialAngles, Object - initial joint angles
   * :return: Object - solved joint angles (best found)
   */
  solve(targetPosition, targetOrientation, initialAngles) {
    let currentAngles = { ...initialAngles };
    let bestAngles = { ...initialAngles };
    let bestError = Infinity;
    const jointNames = this.jointNamesProvider
      ? this.jointNamesProvider()
      : Object.keys(initialAngles);

    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      const iterStart =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      this.robotArm.updateAllJoints(currentAngles);

      const currentPosition = this.robotArm.getEndEffectorPosition();
      const currentOrientation = this.robotArm.getEndEffectorOrientation();

      const positionError = new THREE.Vector3().subVectors(
        targetPosition,
        currentPosition
      );
      const positionErrorMagnitude = positionError.length();

      const orientationError = this.computeOrientationError(
        targetOrientation,
        currentOrientation
      );
      const orientationErrorMagnitude = orientationError.length();

      const totalError =
        positionErrorMagnitude + orientationErrorMagnitude * 0.5;

      if (totalError < bestError) {
        bestError = totalError;
        bestAngles = { ...currentAngles };
      }

      if (
        positionErrorMagnitude < this.positionTolerance &&
        orientationErrorMagnitude < this.orientationTolerance
      ) {
        return currentAngles;
      }

      const J = this.computeGeneralJacobian(currentAngles, jointNames);
      const e = [
        positionError.x,
        positionError.y,
        positionError.z,
        orientationError.x,
        orientationError.y,
        orientationError.z,
      ];

      const lambda = 0.05;
      const delta = this.dampedLeastSquares(J, e, lambda);

      const adaptiveStep = this.computeAdaptiveStepSize(
        positionErrorMagnitude,
        orientationErrorMagnitude
      );
      for (let i = 0; i < jointNames.length; i++) {
        const name = jointNames[i];
        const d = (delta[i] || 0) * adaptiveStep;
        currentAngles[name] = (currentAngles[name] || 0) + d;
        currentAngles[name] = this.applyJointLimits(name, currentAngles[name]);
      }

      if (this.debug) {
        const iterEnd =
          typeof performance !== "undefined" && performance.now
            ? performance.now()
            : Date.now();
        const dt = iterEnd - iterStart;
        try {
          console.log(
            `[IK][LM] iter=${iteration} posErr=${positionErrorMagnitude.toFixed(
              4
            )} ` +
              `oriErr=${orientationErrorMagnitude.toFixed(
                4
              )} total=${totalError.toFixed(4)} ` +
              `lambda=${lambda.toFixed(4)} step=${adaptiveStep.toFixed(
                3
              )} time=${dt.toFixed(2)}ms`
          );
        } catch (_) {}
      }
    }

    return bestAngles;
  }

  /**
   * Build 6×n Jacobian via central differences
   *
   * :param angles, Object - current joint angles
   * :param jointNames, Array<string> - joint names order
   * :return: Array<Array<number>> - 6 by n matrix
   */
  computeGeneralJacobian(angles, jointNames) {
    const epsilon = 0.00005;
    const J = new Array(6)
      .fill(0)
      .map(() => new Array(jointNames.length).fill(0));

    this.robotArm.updateAllJoints(angles);
    const baseOri = this.robotArm.getEndEffectorOrientation();

    for (let j = 0; j < jointNames.length; j++) {
      const jointName = jointNames[j];

      const anglesPos = { ...angles };
      anglesPos[jointName] = (anglesPos[jointName] || 0) + epsilon;
      this.robotArm.updateAllJoints(anglesPos);
      const posPos = this.robotArm.getEndEffectorPosition();
      const oriPos = this.robotArm.getEndEffectorOrientation();

      const anglesNeg = { ...angles };
      anglesNeg[jointName] = (anglesNeg[jointName] || 0) - epsilon;
      this.robotArm.updateAllJoints(anglesNeg);
      const posNeg = this.robotArm.getEndEffectorPosition();
      const oriNeg = this.robotArm.getEndEffectorOrientation();

      const dp = new THREE.Vector3()
        .subVectors(posPos, posNeg)
        .divideScalar(2 * epsilon);

      const w = this.computeAngularVelocity(oriPos, oriNeg, baseOri, epsilon);

      J[0][j] = dp.x;
      J[1][j] = dp.y;
      J[2][j] = dp.z;
      J[3][j] = w.x;
      J[4][j] = w.y;
      J[5][j] = w.z;
    }

    this.robotArm.updateAllJoints(angles);
    return J;
  }

  /**
   * DLS: delta = J^T (J J^T + λ^2 I)^-1 e
   *
   * :param J, Array<Array<number>> - 6×n Jacobian
   * :param e, Array<number> - 6 vector
   * :param lambda, number - damping
   * :return: Array<number> - n vector
   */
  dampedLeastSquares(J, e, lambda) {
    const m = 6;
    const n = J[0].length;
    const JJt = new Array(m).fill(0).map(() => new Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < m; k++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += J[i][j] * J[k][j];
        JJt[i][k] = sum + (i === k ? lambda * lambda : 0);
      }
    }

    const JJtInv = this.invertMatrix(JJt);

    const temp = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let k = 0; k < m; k++) sum += JJtInv[i][k] * e[k];
      temp[i] = sum;
    }

    const delta = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < m; i++) sum += J[i][j] * temp[i];
      delta[j] = sum;
    }
    return delta;
  }

  /**
   * Invert square matrix via Gauss-Jordan
   *
   * :param A, Array<Array<number>> - square matrix
   * :return: Array<Array<number>> - inverse
   */
  invertMatrix(A) {
    const n = A.length;
    const aug = A.map((row, i) => [
      ...row,
      ...row.map((_, j) => (i === j ? 1 : 0)),
    ]);

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let r = i + 1; r < n; r++) {
        if (Math.abs(aug[r][i]) > Math.abs(aug[maxRow][i])) maxRow = r;
      }
      const tmp = aug[i];
      aug[i] = aug[maxRow];
      aug[maxRow] = tmp;
      const pivot = aug[i][i];
      if (Math.abs(pivot) < 1e-12) {
        aug[i][i] = 1e-9;
      }
      const piv = aug[i][i];
      for (let c = 0; c < 2 * n; c++) aug[i][c] /= piv;
      for (let r = 0; r < n; r++) {
        if (r === i) continue;
        const factor = aug[r][i];
        for (let c = 0; c < 2 * n; c++) aug[r][c] -= factor * aug[i][c];
      }
    }

    const inv = new Array(n).fill(0).map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) inv[i][j] = aug[i][j + n];
    return inv;
  }

  /**
   * Compute orientation error as axis-angle vector
   *
   * :param targetQuat, THREE.Quaternion - target
   * :param currentQuat, THREE.Quaternion - current
   * :return: THREE.Vector3 - axis * angle
   */
  computeOrientationError(targetQuat, currentQuat) {
    const relativeQuat = new THREE.Quaternion().multiplyQuaternions(
      targetQuat,
      currentQuat.clone().invert()
    );

    const axis = new THREE.Vector3();
    if (relativeQuat.w < 0) {
      relativeQuat.x *= -1;
      relativeQuat.y *= -1;
      relativeQuat.z *= -1;
      relativeQuat.w *= -1;
    }

    const angle = 2 * Math.acos(Math.abs(relativeQuat.w));

    if (angle > 0.001) {
      const s = Math.sqrt(1 - relativeQuat.w * relativeQuat.w);
      if (s > 0.001) {
        axis.set(relativeQuat.x / s, relativeQuat.y / s, relativeQuat.z / s);
      } else {
        axis.set(1, 0, 0);
      }
      axis.multiplyScalar(angle);
    } else {
      axis.set(0, 0, 0);
    }
    return axis;
  }

  /**
   * Compute angular velocity column via quaternion log map
   *
   * :param orientationPos, THREE.Quaternion - positive perturbation
   * :param orientationNeg, THREE.Quaternion - negative perturbation
   * :param currentOrientation, THREE.Quaternion - base orientation
   * :param epsilon, number - perturbation
   * :return: THREE.Vector3 - angular velocity
   */
  computeAngularVelocity(
    orientationPos,
    orientationNeg,
    currentOrientation,
    epsilon
  ) {
    const relativePos = new THREE.Quaternion().multiplyQuaternions(
      orientationPos,
      currentOrientation.clone().invert()
    );
    const relativeNeg = new THREE.Quaternion().multiplyQuaternions(
      orientationNeg,
      currentOrientation.clone().invert()
    );

    const angleVelPos = this.quaternionToAxisAngle(relativePos);
    const angleVelNeg = this.quaternionToAxisAngle(relativeNeg);

    const angularVelocity = new THREE.Vector3()
      .subVectors(angleVelPos, angleVelNeg)
      .divideScalar(2 * epsilon);
    return angularVelocity;
  }

  /**
   * Quaternion to axis-angle vector
   *
   * :param quaternion, THREE.Quaternion - input q
   * :return: THREE.Vector3 - axis * angle
   */
  quaternionToAxisAngle(quaternion) {
    const q = quaternion.clone();
    if (q.w < 0) {
      q.x *= -1;
      q.y *= -1;
      q.z *= -1;
      q.w *= -1;
    }

    const axisAngle = new THREE.Vector3();
    const angle = 2 * Math.acos(Math.min(1, Math.abs(q.w)));

    if (angle < 1e-6) {
      axisAngle.set(2 * q.x, 2 * q.y, 2 * q.z);
    } else {
      const sinHalfAngle = Math.sqrt(1 - q.w * q.w);
      if (sinHalfAngle > 1e-6) {
        const axis = new THREE.Vector3(q.x, q.y, q.z).divideScalar(
          sinHalfAngle
        );
        axisAngle.copy(axis).multiplyScalar(angle);
      } else {
        axisAngle.set(0, 0, 0);
      }
    }
    return axisAngle;
  }

  /**
   * Adaptive step size from normalized errors
   *
   * :param positionError, number - position error magnitude
   * :param orientationError, number - orientation error magnitude
   * :return: number - step size scale
   */
  computeAdaptiveStepSize(positionError, orientationError) {
    const normalizedPositionError = positionError / 0.01;
    const normalizedOrientationError = orientationError / 0.087;
    const maxNormalizedError = Math.max(
      normalizedPositionError,
      normalizedOrientationError
    );

    if (maxNormalizedError > 2.0) {
      return this.stepSize * 0.8;
    } else if (maxNormalizedError > 1.0) {
      return this.stepSize;
    } else if (maxNormalizedError > 0.5) {
      return this.stepSize * 1.2;
    } else {
      return this.stepSize * 0.6;
    }
  }

  /**
   * Clamp by joint limits
   *
   * :param jointName, string - joint name
   * :param angle, number - angle value
   * :return: number - clamped value
   */
  applyJointLimits(jointName, angle) {
    if (this.jointLimitProvider) {
      const cfg = this.jointLimitProvider(jointName);
      if (cfg && Number.isFinite(cfg.min) && Number.isFinite(cfg.max)) {
        return Math.max(cfg.min, Math.min(cfg.max, angle));
      }
    }
    return angle;
  }
}

/**
 * DLSIKSolver
 *
 * 稳健阻尼最小二乘 IK（SVD + 自适应阻尼），接口与 Advanced6DOFIKSolver 基本一致
 *
 * :param robotArm, Object - robot arm instance providing FK (updateAllJoints, getEndEffectorPosition, getEndEffectorOrientation)
 * :param jointNamesProvider, Function - returns Array<string> of joint names
 * :param jointLimitProvider, Function - returns joint limit { min, max }
 * :return: Object - solver instance with solve(...)
 */
class DLSIKSolver {
  constructor(robotArm, jointNamesProvider, jointLimitProvider) {
    this.robotArm = robotArm;
    this.jointNamesProvider = jointNamesProvider;
    this.jointLimitProvider = jointLimitProvider;
    this.maxIterations = 120;
    this.positionTolerance = 0.003; // 3mm
    this.orientationTolerance = 0.035; // ~2 deg
    this.baseStepSize = 0.035;
    this.minDamping = 1e-4;
    this.maxDamping = 5e-2;
    this.singularValueThreshold = 1e-4; // 小奇异值截断
    this.debug = false; // enable per-iteration logging
    this._lastCond = null; // last condition number estimate
  }

  /**
   * Solve IK using SVD-based damped least squares
   *
   * :param targetPosition, THREE.Vector3 - desired position
   * :param targetOrientation, THREE.Quaternion - desired orientation
   * :param initialAngles, Object - seed joint angles
   * :return: Object - solved joint angles
   */
  solve(targetPosition, targetOrientation, initialAngles) {
    let currentAngles = { ...initialAngles };
    let bestAngles = { ...initialAngles };
    let bestError = Infinity;

    const jointNames = this.jointNamesProvider
      ? this.jointNamesProvider()
      : Object.keys(initialAngles);

    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      const iterStart =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      this.robotArm.updateAllJoints(currentAngles);

      const curPos = this.robotArm.getEndEffectorPosition();
      const curOri = this.robotArm.getEndEffectorOrientation();

      const posErrVec = new THREE.Vector3().subVectors(targetPosition, curPos);
      const oriErrVec = this.computeOrientationError(targetOrientation, curOri);

      const posErr = posErrVec.length();
      const oriErr = oriErrVec.length();
      const totalErr = posErr + 0.5 * oriErr;

      if (totalErr < bestError) {
        bestError = totalErr;
        bestAngles = { ...currentAngles };
      }

      if (
        posErr < this.positionTolerance &&
        oriErr < this.orientationTolerance
      ) {
        return currentAngles;
      }

      const J = this.computeGeneralJacobian(currentAngles, jointNames);
      const e = [
        posErrVec.x,
        posErrVec.y,
        posErrVec.z,
        oriErrVec.x,
        oriErrVec.y,
        oriErrVec.z,
      ];

      const lambda = this.computeAdaptiveDamping(J, posErr, oriErr);
      const delta = this.dlsViaSVD(J, e, lambda);

      const step = this.computeAdaptiveStepSize(posErr, oriErr);
      for (let i = 0; i < jointNames.length; i++) {
        const name = jointNames[i];
        const d = (delta[i] || 0) * step;
        const next = (currentAngles[name] || 0) + d;
        currentAngles[name] = this.applyJointLimits(name, next);
      }

      if (this.debug) {
        const iterEnd =
          typeof performance !== "undefined" && performance.now
            ? performance.now()
            : Date.now();
        const dt = iterEnd - iterStart;
        try {
          console.log(
            `[IK][DLS] iter=${iteration} posErr=${posErr.toFixed(4)} ` +
              `oriErr=${oriErr.toFixed(4)} total=${totalErr.toFixed(4)} ` +
              `lambda=${lambda.toExponential(2)} cond=${(
                this._lastCond || 0
              ).toFixed(1)} ` +
              `step=${step.toFixed(3)} time=${dt.toFixed(2)}ms`
          );
        } catch (_) {}
      }
    }

    return bestAngles;
  }

  /**
   * Build 6×n Jacobian via central differences
   *
   * :param angles, Object - current joint angles
   * :param jointNames, Array<string> - joint names order
   * :return: Array<Array<number>> - 6 by n matrix
   */
  computeGeneralJacobian(angles, jointNames) {
    // 使用FKSolver计算雅可比矩阵
    const fkSolver = this.robotArm.getFKSolver();
    if (fkSolver) {
      return fkSolver.computeJacobian(angles, jointNames);
    }

    // 回退到原始实现
    const epsilon = 0.00005;
    const J = new Array(6)
      .fill(0)
      .map(() => new Array(jointNames.length).fill(0));

    this.robotArm.updateAllJoints(angles);
    const baseOri = this.robotArm.getEndEffectorOrientation();

    for (let j = 0; j < jointNames.length; j++) {
      const jointName = jointNames[j];

      const anglesPos = { ...angles };
      anglesPos[jointName] = (anglesPos[jointName] || 0) + epsilon;
      this.robotArm.updateAllJoints(anglesPos);
      const posPos = this.robotArm.getEndEffectorPosition();
      const oriPos = this.robotArm.getEndEffectorOrientation();

      const anglesNeg = { ...angles };
      anglesNeg[jointName] = (anglesNeg[jointName] || 0) - epsilon;
      this.robotArm.updateAllJoints(anglesNeg);
      const posNeg = this.robotArm.getEndEffectorPosition();
      const oriNeg = this.robotArm.getEndEffectorOrientation();

      const dp = new THREE.Vector3()
        .subVectors(posPos, posNeg)
        .divideScalar(2 * epsilon);
      const w = this.computeAngularVelocity(oriPos, oriNeg, baseOri, epsilon);

      J[0][j] = dp.x;
      J[1][j] = dp.y;
      J[2][j] = dp.z;
      J[3][j] = w.x;
      J[4][j] = w.y;
      J[5][j] = w.z;
    }

    this.robotArm.updateAllJoints(angles);
    return J;
  }

  /**
   * SVD-based damped least squares: delta = V * diag(s / (s^2 + λ^2)) * U^T * e
   *
   * :param J, Array<Array<number>> - 6×n Jacobian
   * :param e, Array<number> - 6 vector
   * :param lambda, number - damping
   * :return: Array<number> - n vector
   */
  dlsViaSVD(J, e, lambda) {
    const { U, S, Vt } = this.svd(J);

    // Compute U^T * e
    const m = U.length;
    const k = S.length;
    const Ut = this.transpose(U);
    const y = new Array(k).fill(0);
    for (let i = 0; i < k; i++) {
      let sum = 0;
      for (let r = 0; r < m; r++) sum += Ut[i][r] * e[r];
      y[i] = sum;
    }

    // Scale by s / (s^2 + lambda^2)
    const z = new Array(k).fill(0);
    const lam2 = lambda * lambda;
    for (let i = 0; i < k; i++) {
      const s = S[i];
      if (s < this.singularValueThreshold) {
        z[i] = 0;
      } else {
        z[i] = (s / (s * s + lam2)) * y[i];
      }
    }

    // delta = V * z, where V = (Vt)^T
    const V = this.transpose(Vt);
    const n = V.length;
    const delta = new Array(n).fill(0);
    for (let r = 0; r < n; r++) {
      let sum = 0;
      for (let i = 0; i < k; i++) sum += V[r][i] * z[i];
      delta[r] = sum;
    }
    return delta;
  }

  /**
   * Thin SVD via Jacobi iterations (sufficient for small 6×n)
   *
   * :param A, Array<Array<number>> - m×n matrix
   * :return: { U, S, Vt }
   */
  svd(A) {
    // Build AtA
    const m = A.length;
    const n = A[0].length;
    const k = Math.min(m, n);
    const At = this.transpose(A);
    const AtA = this.multiply(At, A); // n×n

    // Eigen decomposition (symmetric) for AtA to get V and S^2
    const { eigenVectors: V_full, eigenValues: evalsV_full } = this.eigenSymmetric(AtA);

    // Sort eigenpairs descending by eigenvalue magnitude
    const order = evalsV_full
      .map((v, i) => ({ v: Math.max(v, 0), i }))
      .sort((a, b) => b.v - a.v)
      .map((x) => x.i);

    // Thin SVD components
    const S_full = order.map((i) => Math.sqrt(Math.max(evalsV_full[i], 0)));
    const S = S_full.slice(0, k); // length k

    // Take first k right-singular vectors: V_k (n×k)
    const V_sorted_full = V_full.map((row) => order.map((i) => row[i])); // n×n
    const V_k = V_sorted_full.map((row) => row.slice(0, k)); // n×k
    const Vt = this.transpose(V_k); // k×n

    // U_k = A * V_k * S^{-1}  => (m×n) * (n×k) = (m×k)
    const Sinv = S.map((s) => (s > this.singularValueThreshold ? 1 / s : 0));
    const AV_k = this.multiply(A, V_k); // m×k
    const U = new Array(m).fill(0).map(() => new Array(k).fill(0)); // m×k
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        U[i][j] = AV_k[i][j] * Sinv[j];
      }
    }

    return { U, S, Vt };
  }

  /**
   * Symmetric matrix eigen decomposition via Jacobi method
   *
   * :param M, Array<Array<number>> - n×n symmetric
   * :return: { eigenVectors, eigenValues }
   */
  eigenSymmetric(M) {
    const n = M.length;
    const A = M.map((row) => row.slice());
    const V = this.identity(n);

    const maxIter = 64;
    for (let iter = 0; iter < maxIter; iter++) {
      // Find largest off-diagonal element
      let p = 0,
        q = 1,
        maxVal = Math.abs(A[0][1]);
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const val = Math.abs(A[i][j]);
          if (val > maxVal) {
            maxVal = val;
            p = i;
            q = j;
          }
        }
      }
      if (maxVal < 1e-10) break;

      const app = A[p][p];
      const aqq = A[q][q];
      const apq = A[p][q];
      const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
      const c = Math.cos(phi);
      const s = Math.sin(phi);

      // Rotate A
      for (let i = 0; i < n; i++) {
        const aip = A[i][p];
        const aiq = A[i][q];
        A[i][p] = c * aip - s * aiq;
        A[i][q] = s * aip + c * aiq;
      }
      for (let j = 0; j < n; j++) {
        const apj = A[p][j];
        const aqj = A[q][j];
        A[p][j] = c * apj - s * aqj;
        A[q][j] = s * apj + c * aqj;
      }
      A[p][q] = 0;
      A[q][p] = 0;

      // Update diagonal
      A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;

      // Accumulate V
      for (let i = 0; i < n; i++) {
        const vip = V[i][p];
        const viq = V[i][q];
        V[i][p] = c * vip - s * viq;
        V[i][q] = s * vip + c * viq;
      }
    }

    const eigenValues = new Array(n).fill(0);
    for (let i = 0; i < n; i++) eigenValues[i] = A[i][i];
    return { eigenVectors: V, eigenValues };
  }

  /**
   * Utilities
   */
  transpose(M) {
    return M[0].map((_, i) => M.map((row) => row[i]));
  }
  multiply(A, B) {
    const m = A.length;
    const p = B.length;
    const n = B[0].length;
    const out = new Array(m).fill(0).map(() => new Array(n).fill(0));
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < p; k++) {
        const aik = A[i][k];
        for (let j = 0; j < n; j++) out[i][j] += aik * B[k][j];
      }
    }
    return out;
  }
  identity(n) {
    const I = new Array(n).fill(0).map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) I[i][i] = 1;
    return I;
  }

  /**
   * Compute orientation error as axis-angle vector
   *
   * :param targetQuat, THREE.Quaternion - target
   * :param currentQuat, THREE.Quaternion - current
   * :return: THREE.Vector3 - axis * angle
   */
  computeOrientationError(targetQuat, currentQuat) {
    const relativeQuat = new THREE.Quaternion().multiplyQuaternions(
      targetQuat,
      currentQuat.clone().invert()
    );

    const axis = new THREE.Vector3();
    if (relativeQuat.w < 0) {
      relativeQuat.x *= -1;
      relativeQuat.y *= -1;
      relativeQuat.z *= -1;
      relativeQuat.w *= -1;
    }

    const angle = 2 * Math.acos(Math.abs(relativeQuat.w));

    if (angle > 0.001) {
      const s = Math.sqrt(1 - relativeQuat.w * relativeQuat.w);
      if (s > 0.001) {
        axis.set(relativeQuat.x / s, relativeQuat.y / s, relativeQuat.z / s);
      } else {
        axis.set(1, 0, 0);
      }
      axis.multiplyScalar(angle);
    } else {
      axis.set(0, 0, 0);
    }
    return axis;
  }

  /**
   * Compute angular velocity column via quaternion log map
   *
   * :param orientationPos, THREE.Quaternion - positive perturbation
   * :param orientationNeg, THREE.Quaternion - negative perturbation
   * :param currentOrientation, THREE.Quaternion - base orientation
   * :param epsilon, number - perturbation
   * :return: THREE.Vector3 - angular velocity
   */
  computeAngularVelocity(
    orientationPos,
    orientationNeg,
    currentOrientation,
    epsilon
  ) {
    const relativePos = new THREE.Quaternion().multiplyQuaternions(
      orientationPos,
      currentOrientation.clone().invert()
    );
    const relativeNeg = new THREE.Quaternion().multiplyQuaternions(
      orientationNeg,
      currentOrientation.clone().invert()
    );

    const angleVelPos = this.quaternionToAxisAngle(relativePos);
    const angleVelNeg = this.quaternionToAxisAngle(relativeNeg);

    const angularVelocity = new THREE.Vector3()
      .subVectors(angleVelPos, angleVelNeg)
      .divideScalar(2 * epsilon);
    return angularVelocity;
  }

  /**
   * Quaternion to axis-angle vector
   *
   * :param quaternion, THREE.Quaternion - input q
   * :return: THREE.Vector3 - axis * angle
   */
  quaternionToAxisAngle(quaternion) {
    const q = quaternion.clone();
    if (q.w < 0) {
      q.x *= -1;
      q.y *= -1;
      q.z *= -1;
      q.w *= -1;
    }

    const axisAngle = new THREE.Vector3();
    const angle = 2 * Math.acos(Math.min(1, Math.abs(q.w)));

    if (angle < 1e-6) {
      axisAngle.set(2 * q.x, 2 * q.y, 2 * q.z);
    } else {
      const sinHalfAngle = Math.sqrt(1 - q.w * q.w);
      if (sinHalfAngle > 1e-6) {
        const axis = new THREE.Vector3(q.x, q.y, q.z).divideScalar(
          sinHalfAngle
        );
        axisAngle.copy(axis).multiplyScalar(angle);
      } else {
        axisAngle.set(0, 0, 0);
      }
    }
    return axisAngle;
  }

  /**
   * Adaptive step from normalized errors
   *
   * :param positionError, number
   * :param orientationError, number
   * :return: number
   */
  computeAdaptiveStepSize(positionError, orientationError) {
    const normalizedPositionError = positionError / 0.01;
    const normalizedOrientationError = orientationError / 0.087;
    const maxNormalizedError = Math.max(
      normalizedPositionError,
      normalizedOrientationError
    );
    if (maxNormalizedError > 2.0) return this.baseStepSize * 0.75;
    if (maxNormalizedError > 1.0) return this.baseStepSize;
    if (maxNormalizedError > 0.5) return this.baseStepSize * 1.2;
    return this.baseStepSize * 0.6;
  }

  /**
   * Adaptive damping based on singular values and errors
   *
   * :param J, Array<Array<number>> - Jacobian
   * :param posErr, number
   * :param oriErr, number
   * :return: number - lambda
   */
  computeAdaptiveDamping(J, posErr, oriErr) {
    // Estimate condition via SVD on-demand (cheap for 6×n)
    const { S } = this.svd(J);
    const sMax = Math.max(...S, this.singularValueThreshold);
    const sMin = Math.max(Math.min(...S), this.singularValueThreshold);
    const cond = sMax / sMin;
    this._lastCond = cond;

    // Base on error magnitude
    const err = posErr + 0.5 * oriErr;
    let lambda = this.minDamping + (this.maxDamping - this.minDamping) * 0.5;
    if (cond > 200 || err > 0.05) lambda = this.maxDamping;
    else if (cond < 30 && err < 0.01) lambda = this.minDamping;
    return lambda;
  }

  /**
   * Clamp by joint limits
   *
   * :param jointName, string
   * :param angle, number
   * :return: number
   */
  applyJointLimits(jointName, angle) {
    if (this.jointLimitProvider) {
      const cfg = this.jointLimitProvider(jointName);
      if (cfg && Number.isFinite(cfg.min) && Number.isFinite(cfg.max)) {
        return Math.max(cfg.min, Math.min(cfg.max, angle));
      }
    }
    return angle;
  }
}

// UMD-like export for browser global or CommonJS
if (typeof module !== "undefined" && module.exports) {
  module.exports = { JacobianLM6DoFIKSolver, DLSIKSolver };
} else {
  window.JacobianLM6DoFIKSolver = JacobianLM6DoFIKSolver;
  window.DLSIKSolver = DLSIKSolver;
}


