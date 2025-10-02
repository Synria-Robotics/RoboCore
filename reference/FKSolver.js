/**
 * 正运动学求解器
 * Forward Kinematics Solver
 */
class FKSolver {
  constructor(robotArm) {
    this.robotArm = robotArm;
  }

  /**
   * 从关节层次结构动态构建运动链
   * @param {Object} jointHierarchy - 关节层次结构信息
   * @returns {Array<string>} 运动链数组
   */
  buildKinematicChain(jointHierarchy) {
    const kinematicChain = [];
    
    // 方法1: 基于层次结构的深度优先遍历
    const buildChainFromHierarchy = (startNode, visited = new Set()) => {
      if (!startNode || visited.has(startNode)) return;
      
      visited.add(startNode);
      
      // 如果是关节（有type属性），添加到运动链
      const nodeInfo = jointHierarchy[startNode];
      if (nodeInfo && nodeInfo.type) {
        kinematicChain.push(startNode);
      }
      
      // 递归处理子节点
      if (nodeInfo && nodeInfo.children) {
        for (const child of nodeInfo.children) {
          buildChainFromHierarchy(child, visited);
        }
      }
    };
    
    // 从根节点开始构建运动链
    const rootNodes = Object.keys(jointHierarchy).filter(name => {
      const node = jointHierarchy[name];
      return node && !node.parent; // 没有父节点的就是根节点
    });
    
    for (const rootNode of rootNodes) {
      buildChainFromHierarchy(rootNode);
    }
    
    // 方法2: 如果层次结构方法失败，回退到模式匹配
    if (kinematicChain.length === 0) {
      const jointPatterns = [
        /^joint\d+$/i,           // joint1, joint2, etc.
        /^Joint\d+$/i,           // Joint1, Joint2, etc.
        /^j\d+$/i,               // j1, j2, etc.
        /^link\d+$/i,            // link1, link2, etc.
        /^revolute\d+$/i,        // revolute1, revolute2, etc.
      ];
      
      // 按数字顺序排序的关节
      const sortedJoints = Object.keys(jointHierarchy)
        .filter(name => {
          // 检查是否匹配任何关节模式
          return jointPatterns.some(pattern => pattern.test(name));
        })
        .sort((a, b) => {
          // 提取数字进行排序
          const numA = parseInt(a.match(/\d+/)?.[0] || '0');
          const numB = parseInt(b.match(/\d+/)?.[0] || '0');
          return numA - numB;
        });
      
      kinematicChain.push(...sortedJoints);
    }
    
    // 添加工具端点（如果存在且未包含在运动链中）
    const toolNames = ['tool0', 'tool', 'end_effector', 'gripper', 'flange'];
    for (const toolName of toolNames) {
      if (jointHierarchy[toolName] && !kinematicChain.includes(toolName)) {
        kinematicChain.push(toolName);
        break;
      }
    }
    
    // 调试信息
    if (console && console.log) {
      console.log('FKSolver: 构建的运动链:', kinematicChain);
      console.log('FKSolver: 可用的关节层次结构:', Object.keys(jointHierarchy));
    }
    
    return kinematicChain;
  }

  /**
   * 计算指定关节角度下的正向运动学 (精确迭代实现)
   * @param {Object} jointAngles - 关节角度集合 { jointName: angle, ... }
   * @param {Object} jointHierarchy - 关节层次结构信息（可选，如果不提供则从robotArm获取）
   * @returns {Object} 包含世界坐标系下位置和姿态的对象 { position: THREE.Vector3, orientation: THREE.Quaternion }
   */
  calculateForwardKinematics(jointAngles, jointHierarchy = null) {
    // 获取关节层次结构
    const hierarchy = jointHierarchy || (this.robotArm ? this.robotArm.jointHierarchy : {});
    
    // 动态构建运动链，从关节层次结构中提取
    const kinematicChain = this.buildKinematicChain(hierarchy);

    // 初始变换：应用从Z-up到Y-up的根旋转
    const rootTransform = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
    let finalTransform = new THREE.Matrix4().copy(rootTransform);

    // 迭代计算链条上每个关节的变换
    kinematicChain.forEach((partName) => {
      const partInfo = jointHierarchy[partName];
      if (!partInfo) return;

      const localTransform = new THREE.Matrix4();
      const origin = partInfo.origin || { x: 0, y: 0, z: 0 };
      const rotation = partInfo.rotation || { x: 0, y: 0, z: 0 };

      // 构造局部的静态变换矩阵 (T * R)
      const staticRotationMatrix = new THREE.Matrix4();
      if (rotation.x !== 0)
        staticRotationMatrix.multiply(
          new THREE.Matrix4().makeRotationX(rotation.x)
        );
      if (rotation.y !== 0)
        staticRotationMatrix.multiply(
          new THREE.Matrix4().makeRotationY(rotation.y)
        );
      if (rotation.z !== 0)
        staticRotationMatrix.multiply(
          new THREE.Matrix4().makeRotationZ(rotation.z)
        );

      const staticTransform = new THREE.Matrix4()
        .makeTranslation(origin.x, origin.y, origin.z)
        .multiply(staticRotationMatrix);

      localTransform.copy(staticTransform);

      // 应用动态的关节旋转 (如果是可动关节)
      if (
        partInfo.type === "revolute" &&
        jointAngles.hasOwnProperty(partName)
      ) {
        const angle = jointAngles[partName];
        const axis = new THREE.Vector3(
          partInfo.axis.x,
          partInfo.axis.y,
          partInfo.axis.z
        );
        const jointRotationMatrix = new THREE.Matrix4().makeRotationAxis(
          axis,
          angle
        );
        localTransform.multiply(jointRotationMatrix);
      }

      // 将局部变换累积到最终变换中
      finalTransform.multiply(localTransform);
    });

    const position = new THREE.Vector3();
    const orientation = new THREE.Quaternion();
    const scale = new THREE.Vector3();

    finalTransform.decompose(position, orientation, scale);

    return { position, orientation };
  }

  /**
   * 通过场景图计算正运动学 (基于Three.js场景图)
   * @param {Object} jointAngles - 关节角度集合
   * @returns {Object} 包含位置和姿态的对象
   */
  calculateForwardKinematicsFromScene(jointAngles) {
    if (!this.robotArm || !this.robotArm.isLoaded) {
      return { position: new THREE.Vector3(), orientation: new THREE.Quaternion() };
    }

    // 更新所有关节角度
    this.robotArm.updateAllJoints(jointAngles);

    // 获取末端执行器位置和方向
    const position = this.robotArm.getEndEffectorPosition();
    const orientation = this.robotArm.getEndEffectorOrientation();

    return { position, orientation };
  }

  /**
   * 计算雅可比矩阵 (用于IK求解)
   * @param {Object} jointAngles - 当前关节角度
   * @param {Array} jointNames - 关节名称数组
   * @returns {Array<Array<number>>} 6×n雅可比矩阵
   */
  computeJacobian(jointAngles, jointNames) {
    const epsilon = 0.00005;
    const J = new Array(6)
      .fill(0)
      .map(() => new Array(jointNames.length).fill(0));

    // 获取当前末端执行器位姿
    const currentPose = this.calculateForwardKinematicsFromScene(jointAngles);
    const baseOri = currentPose.orientation;

    for (let j = 0; j < jointNames.length; j++) {
      const jointName = jointNames[j];

      // 正向扰动
      const anglesPos = { ...jointAngles };
      anglesPos[jointName] = (anglesPos[jointName] || 0) + epsilon;
      const posPose = this.calculateForwardKinematicsFromScene(anglesPos);

      // 负向扰动
      const anglesNeg = { ...jointAngles };
      anglesNeg[jointName] = (anglesNeg[jointName] || 0) - epsilon;
      const negPose = this.calculateForwardKinematicsFromScene(anglesNeg);

      // 计算位置导数
      const dp = new THREE.Vector3()
        .subVectors(posPose.position, negPose.position)
        .divideScalar(2 * epsilon);

      // 计算角速度
      const w = this.computeAngularVelocity(
        posPose.orientation,
        negPose.orientation,
        baseOri,
        epsilon
      );

      // 填充雅可比矩阵
      J[0][j] = dp.x;
      J[1][j] = dp.y;
      J[2][j] = dp.z;
      J[3][j] = w.x;
      J[4][j] = w.y;
      J[5][j] = w.z;
    }

    return J;
  }

  /**
   * 计算角速度
   * @param {THREE.Quaternion} oriPos - 正向扰动后的方向
   * @param {THREE.Quaternion} oriNeg - 负向扰动后的方向
   * @param {THREE.Quaternion} baseOri - 基础方向
   * @param {number} epsilon - 扰动大小
   * @returns {THREE.Vector3} 角速度向量
   */
  computeAngularVelocity(oriPos, oriNeg, baseOri, epsilon) {
    const qPos = oriPos.clone().multiply(baseOri.clone().invert());
    const qNeg = oriNeg.clone().multiply(baseOri.clone().invert());
    
    const qDiff = qPos.clone().multiply(qNeg.clone().invert());
    const axis = new THREE.Vector3();
    const angle = qDiff.getAxisAngle(axis);
    
    return axis.multiplyScalar(angle / (2 * epsilon));
  }

  /**
   * 验证关节角度是否在限制范围内
   * @param {Object} jointAngles - 关节角度集合
   * @param {Object} jointLimits - 关节限制信息
   * @returns {boolean} 是否在限制范围内
   */
  validateJointLimits(jointAngles, jointLimits) {
    for (const [jointName, angle] of Object.entries(jointAngles)) {
      const limits = jointLimits[jointName];
      if (limits) {
        if (angle < limits.lower || angle > limits.upper) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * 计算末端执行器到目标点的距离
   * @param {Object} jointAngles - 关节角度集合
   * @param {THREE.Vector3} targetPosition - 目标位置
   * @returns {number} 距离
   */
  calculatePositionError(jointAngles, targetPosition) {
    const pose = this.calculateForwardKinematicsFromScene(jointAngles);
    return pose.position.distanceTo(targetPosition);
  }

  /**
   * 计算末端执行器到目标姿态的角度误差
   * @param {Object} jointAngles - 关节角度集合
   * @param {THREE.Quaternion} targetOrientation - 目标姿态
   * @returns {number} 角度误差 (弧度)
   */
  calculateOrientationError(jointAngles, targetOrientation) {
    const pose = this.calculateForwardKinematicsFromScene(jointAngles);
    const quatDiff = pose.orientation.clone().multiply(targetOrientation.clone().invert());
    return 2 * Math.acos(Math.abs(quatDiff.w));
  }
}

// 导出类
if (typeof module !== 'undefined' && module.exports) {
  module.exports = FKSolver;
} else if (typeof window !== 'undefined') {
  window.FKSolver = FKSolver;
}
