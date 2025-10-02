/**
 * 机械臂3D模型类
 * 负责加载STL文件、构建关节层次结构和实现运动学
 */
class RobotArm {
  constructor(scene) {
    this.scene = scene;
    this.loader = new THREE.STLLoader();
    this.textureLoader = new THREE.TextureLoader();
    this.meshes = {};
    this.joints = {};
    this.loadingPromises = [];
    this.isLoaded = false;
  this.coordinateFrames = new Map(); // 旧：统一存储
  this.bodyFrames = new Map(); // link(body) 坐标系
  this.jointFrames = new Map(); // joint 坐标系
  this.coordinateFramesMode = 'none'; // none | body | joint
  this.showCoordinateFrames = false; // 兼容旧逻辑

    // 正运动学求解器
    this.fkSolver = null;

    // 机械臂根节点
    this.robotGroup = new THREE.Group();
    this.robotGroup.name = "robot_arm";

    // Z-up坐标系，不需要额外的旋转转换
    this.scene.add(this.robotGroup);

    // 定义关节层次结构（根据URDF文件）
    this.jointHierarchy = {
      base_link: {
        mesh: "base_link.STL",
        children: ["joint1"],
      },
      joint1: {
        parent: "base_link",
        origin: { x: 0, y: 0, z: 0.1445 },
        rotation: { x: 0, y: 0, z: 0 },
        axis: { x: 0, y: 0, z: 1 },
        type: "revolute",
        children: ["link1"],
      },
      link1: {
        parent: "joint1",
        mesh: "Link1.STL",
        children: ["joint2"],
      },
      joint2: {
        parent: "link1",
        origin: { x: 0, y: 0, z: 0.025106 },
        rotation: { x: 1.5708, y: -1.5708, z: 3.14 },
        axis: { x: 0, y: 0, z: -1 },
        type: "revolute",
        children: ["link2"],
      },
      link2: {
        parent: "joint2",
        mesh: "Link2.STL",
        children: ["joint3"],
      },
      joint3: {
        parent: "link2",
        origin: { x: 0.22367, y: 0.022494, z: -0.00005 },
        rotation: { x: 0, y: 0, z: 2.3562 },
        axis: { x: 0, y: 0, z: -1 },
        type: "revolute",
        children: ["link3"],
      },
      link3: {
        parent: "joint3",
        mesh: "Link3.STL",
        children: ["joint4"],
      },
      joint4: {
        parent: "link3",
        origin: { x: 0.0988, y: 0.00211, z: -0.0001 },
        rotation: { x: 1.5708, y: 0, z: 1.5708 },
        axis: { x: 0, y: 0, z: -1 },
        type: "revolute",
        children: ["link4"],
      },
      link4: {
        parent: "joint4",
        mesh: "Link4.STL",
        children: ["joint5"],
      },
      joint5: {
        parent: "link4",
        origin: { x: 0, y: -0.0007, z: 0.12011 },
        rotation: { x: -1.5708, y: 0, z: 0 },
        axis: { x: 0, y: 0, z: -1 },
        type: "revolute",
        children: ["link5"],
      },
      link5: {
        parent: "joint5",
        mesh: "Link5.STL",
        children: ["joint6"],
      },
      joint6: {
        parent: "link5",
        origin: { x: -0.0038938, y: -0.0573, z: 0.0008 },
        rotation: { x: 1.5708, y: 0, z: 0 },
        axis: { x: 0, y: 0, z: -1 },
        type: "revolute",
        children: ["link6", "grasp_base"],
      },
      link6: {
        parent: "joint6",
        mesh: "Link6.STL",
      },
      grasp_base: {
        parent: "joint6",
        mesh: "Grasp_base.STL",
        children: ["left_finger", "right_finger", "tool0"],
      },
      left_finger: {
        parent: "grasp_base",
        origin: { x: 0.00275, y: -0.0011661, z: 0.13779 },
        rotation: { x: 1.5451, y: 0, z: 0 },
        axis: { x: 0, y: 0, z: -1 },
        type: "prismatic",
        children: ["link7"],
      },
      link7: {
        parent: "left_finger",
        mesh: "Link7.STL",
      },
      right_finger: {
        parent: "grasp_base",
        origin: { x: 0.00275, y: 0.0028325, z: 0.13768 },
        rotation: { x: -1.5965, y: 0, z: 0 },
        axis: { x: 0, y: 0, z: 1 },
        type: "prismatic",
        children: ["link8"],
      },
      link8: {
        parent: "right_finger",
        mesh: "Link8.STL",
      },
      tool0: {
        parent: "grasp_base",
        origin: { x: 0.00275, y: 0.0008332, z: 0.13779 },
        rotation: { x: 0, y: 0, z: 0 },
        type: "fixed",
        isVirtual: true,
      },
    };

    // 材质定义
    this.materials = {
      default: new THREE.MeshLambertMaterial({
        color: 0xcccccc,
        side: THREE.DoubleSide,
      }),
      highlight: new THREE.MeshLambertMaterial({
        color: 0x4caf50,
        side: THREE.DoubleSide,
      }),
      gripper: new THREE.MeshLambertMaterial({
        color: 0x2196f3,
        side: THREE.DoubleSide,
      }),
      tool: new THREE.MeshBasicMaterial({
        color: 0x00ff00,
        transparent: true,
        opacity: 0.8,
      }),
    };
  }

  /**
   * 计算指定关节角度下的正向运动学 (精确迭代实现)
   * @param {Object} jointAngles - 关节角度集合 { jointName: angle, ... }
   * @returns {Object} 包含世界坐标系下位置和姿态的对象 { position: THREE.Vector3, orientation: THREE.Quaternion }
   */
  calculateForwardKinematics(jointAngles) {
    if (this.fkSolver) {
      return this.fkSolver.calculateForwardKinematics(jointAngles);
    }
    
    // 回退到基于场景图的计算
    return this.fkSolver ? this.fkSolver.calculateForwardKinematicsFromScene(jointAngles) : 
           { position: new THREE.Vector3(), orientation: new THREE.Quaternion() };
  }

  /**
   * 加载机械臂模型
   * @returns {Promise} 加载完成的Promise
   */
  async loadRobotArm() {
    try {
      // 加载所有STL文件
      const stlFiles = [
        "base_link.STL",
        "Link1.STL",
        "Link2.STL",
        "Link3.STL",
        "Link4.STL",
        "Link5.STL",
        "Link6.STL",
        "Link7.STL",
        "Link8.STL",
        "Grasp_base.STL",
      ];

      const loadPromises = stlFiles.map((file) => this.loadSTL(file));
      await Promise.all(loadPromises);

      // 构建机械臂层次结构
      this.buildRobotHierarchy();

      // 添加工具端点可视化
      this.addToolVisualization();

      this.isLoaded = true;
      
      // 初始化正运动学求解器
      this.fkSolver = new FKSolver(this);
      
      // 初始化坐标系可视化
      this.initializeCoordinateFrames();
      
      console.log("机械臂模型加载完成");
    } catch (error) {
      console.error("加载机械臂模型失败:", error);
      throw error;
    }
  }

  /**
   * 加载STL文件
   * @param {string} filename - STL文件名
   * @returns {Promise} 加载完成的Promise
   */
  loadSTL(filename) {
    return new Promise((resolve, reject) => {
      this.loader.load(
        `assets/urdf/Alicia_D_v5_4/meshes/${filename}`,
        (geometry) => {
          // Ensure smooth shading lighting by computing normals
          if (geometry && geometry.computeVertexNormals) {
            geometry.computeVertexNormals();
          }
          // 居中几何体 - 移除此行来保证原始坐标
          // geometry.center();

          // 创建网格
          const material = this.getMaterialForPart(filename);
          const mesh = new THREE.Mesh(geometry, material);
          mesh.name = filename;

          // 添加阴影
          mesh.castShadow = true;
          mesh.receiveShadow = true;

          this.meshes[filename] = mesh;
          resolve(mesh);
        },
        (progress) => {
          console.log(
            `加载进度 ${filename}: ${(progress.loaded / progress.total) * 100}%`
          );
        },
        (error) => {
          console.error(`加载失败 ${filename}:`, error);
          reject(error);
        }
      );
    });
  }

  /**
   * 从URDF数据加载机械臂（层级与mesh）
   *
   * :param robotData, Object - URDFParser.parseURDF 返回的对象
   * :return: Promise<void>
   */
  async loadFromURDF(robotData) {
    // 清空旧内容
    this.dispose();

    // 重新创建根组
    this.robotGroup = new THREE.Group();
    this.robotGroup.name = "robot_arm";
    this.scene.add(this.robotGroup);

    this.meshes = {};
    this.joints = {};
    this.isLoaded = false;
    this.coordinateFrames = new Map(); // 存储坐标系可视化对象
    this.showCoordinateFrames = false;

    // 重新初始化材质，避免使用已dispose的材质
    this.materials = {
      default: new THREE.MeshLambertMaterial({
        color: 0xcccccc,
        side: THREE.DoubleSide,
      }),
      highlight: new THREE.MeshLambertMaterial({
        color: 0x4caf50,
        side: THREE.DoubleSide,
      }),
      gripper: new THREE.MeshLambertMaterial({
        color: 0x2196f3,
        side: THREE.DoubleSide,
      }),
      tool: new THREE.MeshBasicMaterial({
        color: 0x00ff00,
        transparent: true,
        opacity: 0.8,
      }),
    };

    const baseDir = robotData?.baseDir || "";
    const links = robotData?.links || {};
    const jointsMap = robotData?.joints || {};

    // 持久化URDF图以便候选分析
    this.urdfLinks = Object.keys(links);
    this.urdfJointsMap = jointsMap;

    // 1) 先为每个link创建一个Group，并在有mesh时加载mesh
    for (const linkName of Object.keys(links)) {
      const linkInfo = links[linkName];
      const group = new THREE.Group();
      group.name = linkName;
      this.joints[linkName] = group;

      // visual meshes (support multiple visuals from MJCF or single from URDF)
      const visuals = Array.isArray(linkInfo.visuals)
        ? linkInfo.visuals
        : linkInfo.visual
        ? [linkInfo.visual]
        : [];
      for (const v of visuals) {
        const meshInfo = v && v.geometry;
        if (meshInfo && meshInfo.type === "mesh" && meshInfo.filename) {
          try {
            const mesh = await this.loadSTLFromPath(meshInfo.filename);
            // per-visual scale
            if (Array.isArray(meshInfo.scale) && meshInfo.scale.length === 3) {
              mesh.scale.set(
                meshInfo.scale[0],
                meshInfo.scale[1],
                meshInfo.scale[2]
              );
            }
            // per-visual material (URDF/MJCF)
            const vMat = v.material;
            if (vMat) {
              const material = this.createMaterialFromURDF(vMat);
              mesh.material = material;
              mesh.userData.originalMaterial = material;
            } else {
              mesh.userData.originalMaterial = mesh.material;
            }
            // per-visual origin
            const vOrigin = v.origin;
            if (vOrigin) {
              const vpos = vOrigin.xyz || [0, 0, 0];
              mesh.position.set(vpos[0], vpos[1], vpos[2]);
              const vrpy = vOrigin.rpy || [0, 0, 0];
              const veuler = new THREE.Euler(vrpy[0], vrpy[1], vrpy[2], "ZYX");
              mesh.quaternion.setFromEuler(veuler);
            }
            group.add(mesh);
          } catch (e) {
            console.warn(`加载mesh失败: ${meshInfo.filename}`, e);
          }
        } else if (meshInfo && meshInfo.type && meshInfo.type !== "mesh") {
          try {
            // Create primitive geometry mesh for box/sphere/cylinder/capsule
            const material = v.material
              ? this.createMaterialFromURDF(v.material)
              : this.materials.default;
            const prim = this.createPrimitiveMeshFromURDFGeometry(
              meshInfo,
              material
            );
            if (prim) {
              // Apply origin (position + rpy)
              const vOrigin = v.origin || { xyz: [0, 0, 0], rpy: [0, 0, 0] };
              const vpos = vOrigin.xyz || [0, 0, 0];
              prim.position.set(vpos[0], vpos[1], vpos[2]);
              const vrpy = vOrigin.rpy || [0, 0, 0];
              const veuler = new THREE.Euler(vrpy[0], vrpy[1], vrpy[2], "ZYX");
              prim.quaternion.setFromEuler(veuler);
              prim.userData.originalMaterial = prim.material;
              group.add(prim);
            }
          } catch (e) {
            console.warn("创建基础形状失败:", meshInfo, e);
          }
        }
      }
    }

    // 将根link附加到robotGroup
    const childLinkSet = new Set();
    Object.values(jointsMap).forEach((j) => childLinkSet.add(j.child));
    Object.keys(links).forEach((linkName) => {
      if (!childLinkSet.has(linkName)) {
        const rootLinkNode = this.joints[linkName];
        if (rootLinkNode) this.robotGroup.add(rootLinkNode);
      }
    });

    // 2) 为每个joint创建一个Group（动作在joint节点上），并设置origin和轴
    const jointNodes = {};
    for (const jointName of Object.keys(jointsMap)) {
      const j = jointsMap[jointName];
      const jointGroup = new THREE.Group();
      jointGroup.name = jointName;

      // 记录静态旋转（URDF origin rpy）
      if (j.origin) {
        const euler = new THREE.Euler(
          j.origin.rpy[0] || 0,
          j.origin.rpy[1] || 0,
          j.origin.rpy[2] || 0,
          "ZYX"
        );
        jointGroup.userData.initialQuaternion =
          new THREE.Quaternion().setFromEuler(euler);
        const pos = j.origin.xyz || [0, 0, 0];
        jointGroup.position.set(pos[0], pos[1], pos[2]);
      } else {
        jointGroup.userData.initialQuaternion = new THREE.Quaternion();
      }

      // 关节轴
      const axis = j.axis || [0, 0, 1];
      jointGroup.userData.axis = new THREE.Vector3(axis[0], axis[1], axis[2]);
      jointGroup.userData.jointType = j.type;

      // 平移关节的轴组
      if (j.type === "prismatic") {
        const axisGroup = new THREE.Group();
        axisGroup.name = jointName + "_axis";
        jointGroup.add(axisGroup);
        jointGroup.userData.axisGroup = axisGroup;
      }

      this.joints[jointName] = jointGroup;
      jointNodes[jointName] = jointGroup;
    }

    // 3) 按照URDF父子关系组装：parent link -> joint -> child link
    for (const jointName of Object.keys(jointsMap)) {
      const j = jointsMap[jointName];
      const parentLinkNode = this.joints[j.parent];
      const childLinkNode = this.joints[j.child];
      const jointNode = jointNodes[jointName];

      const parentNode = parentLinkNode ? parentLinkNode : this.robotGroup;

      // 平移关节父节点的axisGroup处理
      if (
        parentNode &&
        parentNode.userData &&
        parentNode.userData.jointType === "prismatic" &&
        parentNode.userData.axisGroup
      ) {
        parentNode.userData.axisGroup.add(jointNode);
      } else {
        parentNode.add(jointNode);
      }

      if (childLinkNode) {
        // joint -> child link，同样考虑joint是prismatic时，有axisGroup
        if (
          jointNode.userData.jointType === "prismatic" &&
          jointNode.userData.axisGroup
        ) {
          jointNode.userData.axisGroup.add(childLinkNode);
        } else {
          jointNode.add(childLinkNode);
        }
      }

      // 静态姿态
      if (jointNode.userData.initialQuaternion) {
        jointNode.quaternion.copy(jointNode.userData.initialQuaternion);
      }
    }

    // 4) 如果有 tool0 或 URDF中指定的末端，绑定工具端点
    this.toolEndEffector = null;
    if (this.joints["tool0"]) {
      // 如果有显式tool0，用它的世界位置
      const placeholder = new THREE.Object3D();
      placeholder.name = "tool_sphere";
      placeholder.position.set(0, 0, 0);
      this.joints["tool0"].add(placeholder);
      this.toolEndEffector = placeholder;
    } else {
      // fallback: 取最后的子节点（启发式）
      const last = this.findLeafNode();
      if (last) {
        const placeholder = new THREE.Object3D();
        placeholder.name = "tool_sphere";
        placeholder.position.set(0, 0, 0);
        last.add(placeholder);
        this.toolEndEffector = placeholder;
      }
    }

    this.isLoaded = true;

    // 初始化坐标系可视化
    this.initializeCoordinateFrames();
  }

  /**
   * Create Three.js mesh from URDF/MJCF primitive geometry
   *
   * :param geom, Object, geometry description {type, ...}
   * :param material, THREE.Material
   * :return: THREE.Mesh|null
   */
  createPrimitiveMeshFromURDFGeometry(geom, material) {
    if (!geom || !geom.type) return null;
    const t = String(geom.type).toLowerCase();
    let mesh = null;

    if (t === "box") {
      const size = Array.isArray(geom.size) ? geom.size : [0.1, 0.1, 0.1];
      const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
      mesh = new THREE.Mesh(g, material);
    } else if (t === "sphere") {
      const radius = Number.isFinite(geom.radius) ? geom.radius : 0.05;
      const g = new THREE.SphereGeometry(radius, 24, 16);
      mesh = new THREE.Mesh(g, material);
    } else if (t === "cylinder") {
      const radius = Number.isFinite(geom.radius) ? geom.radius : 0.02;
      const length = Number.isFinite(geom.length) ? geom.length : 0.1;
      // Three.js CylinderGeometry is aligned with +Y
      const g = new THREE.CylinderGeometry(radius, radius, length, 24);
      mesh = new THREE.Mesh(g, material);
    } else if (t === "capsule") {
      const radius = Number.isFinite(geom.radius) ? geom.radius : 0.02;
      const tipToTip = Number.isFinite(geom.length) ? geom.length : 0.14;
      const cylinderLength = Math.max(0, tipToTip - 2 * radius);
      // CapsuleGeometry(length along Y)
      if (THREE.CapsuleGeometry) {
        const g = new THREE.CapsuleGeometry(radius, cylinderLength, 16, 12);
        mesh = new THREE.Mesh(g, material);
      } else {
        // Fallback: approximate capsule by cylinder + spheres
        const group = new THREE.Group();
        const cyl = new THREE.Mesh(
          new THREE.CylinderGeometry(radius, radius, cylinderLength, 24),
          material
        );
        const sphTop = new THREE.Mesh(
          new THREE.SphereGeometry(radius, 24, 16),
          material
        );
        const sphBot = new THREE.Mesh(
          new THREE.SphereGeometry(radius, 24, 16),
          material
        );
        sphTop.position.y = cylinderLength / 2;
        sphBot.position.y = -cylinderLength / 2;
        group.add(cyl);
        group.add(sphTop);
        group.add(sphBot);
        return group; // return group instead of single mesh
      }
    }

    if (mesh) {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
    return mesh;
  }

  /**
   * Create Three.js material from URDF material description
   *
   * :param urdfMaterial, Object - {name, color:[r,g,b,a], texture:string|null}
   * :return: THREE.Material
   */
  createMaterialFromURDF(urdfMaterial) {
    const rgba = Array.isArray(urdfMaterial.color)
      ? urdfMaterial.color
      : [0.8, 0.8, 0.8, 1.0];

    // URDF 颜色通常按 sRGB 定义，这里转换到线性空间以匹配 three.js 工作色域
    const color = new THREE.Color()
      .setRGB(rgba[0], rgba[1], rgba[2])
      .convertSRGBToLinear();
    const alpha = typeof rgba[3] === "number" ? rgba[3] : 1.0;

    const material = new THREE.MeshLambertMaterial({
      color: color,
      side: THREE.DoubleSide,
      transparent: alpha < 1.0,
      opacity: alpha,
    });

    if (urdfMaterial.texture) {
      // Load texture asynchronously and assign once available
      this.textureLoader.load(
        urdfMaterial.texture,
        (tex) => {
          try {
            if (tex) {
              tex.colorSpace = THREE.SRGBColorSpace;
              material.map = tex;
              material.needsUpdate = true;
            }
          } catch (err) {
            console.warn("应用纹理失败:", err);
          }
        },
        undefined,
        (err) => {
          console.warn("纹理加载失败:", urdfMaterial.texture, err);
        }
      );
    }

    return material;
  }

  /**
   * 通过绝对/相对路径加载STL
   * @param {string} fullPath - STL路径
   * @returns {Promise<THREE.Mesh>}
   */
  loadSTLFromPath(fullPath) {
    return new Promise((resolve, reject) => {
      this.loader.load(
        fullPath,
        (geometry) => {
          // Ensure smooth shading lighting by computing normals
          if (geometry && geometry.computeVertexNormals) {
            geometry.computeVertexNormals();
          }
          const material = this.materials.default;
          const mesh = new THREE.Mesh(geometry, material);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          resolve(mesh);
        },
        undefined,
        (err) => reject(err)
      );
    });
  }

  /**
   * 查找一个叶子节点（没有子元素的link）
   * @returns {THREE.Object3D|null}
   */
  findLeafNode() {
    let leaf = null;
    const traverse = (node) => {
      if (node.children.length === 0) {
        leaf = node;
        return;
      }
      node.children.forEach(traverse);
    };
    traverse(this.robotGroup);
    return leaf;
  }

  /**
   * 根据部件名称获取材质
   * @param {string} filename - 文件名
   * @returns {THREE.Material} 材质
   */
  getMaterialForPart(filename) {
    if (filename.includes("Link7") || filename.includes("Link8")) {
      return this.materials.gripper;
    } else if (filename.includes("Grasp_base")) {
      return this.materials.highlight;
    } else {
      return this.materials.default;
    }
  }

  /**
   * 构建机械臂层次结构
   */
  buildRobotHierarchy() {
    // 创建所有节点
    Object.keys(this.jointHierarchy).forEach((name) => {
      const component = this.jointHierarchy[name];

      if (component.mesh && this.meshes[component.mesh]) {
        const group = new THREE.Group();
        group.name = name;
        group.add(this.meshes[component.mesh]);
        this.joints[name] = group;
      } else if (component.type) {
        const group = new THREE.Group();
        group.name = name;
        this.joints[name] = group;

        // 为平移关节创建一个额外的轴节点，以修正变换顺序
        if (component.type === "prismatic") {
          const axisGroup = new THREE.Group();
          axisGroup.name = name + "_axis";
          group.add(axisGroup);
          group.userData.axisGroup = axisGroup;
        }

        if (component.rotation) {
          const euler = new THREE.Euler(
            component.rotation.x,
            component.rotation.y,
            component.rotation.z,
            "ZYX" // URDF rpy entspricht 'ZYX' Euler-Ordnung
          );
          group.userData.initialQuaternion =
            new THREE.Quaternion().setFromEuler(euler);
        } else {
          group.userData.initialQuaternion = new THREE.Quaternion();
        }
      }
    });

    // 建立父子关系
    Object.keys(this.jointHierarchy).forEach((name) => {
      const component = this.jointHierarchy[name];
      const node = this.joints[name];

      const parentNode =
        component.parent && this.joints[component.parent]
          ? this.joints[component.parent]
          : this.robotGroup;

      // 如果父节点是平移关节，则添加到其axisGroup中
      const parentComponent = this.jointHierarchy[component.parent];
      if (
        parentComponent &&
        parentComponent.type === "prismatic" &&
        parentNode.userData.axisGroup
      ) {
        parentNode.userData.axisGroup.add(node);
      } else {
        parentNode.add(node);
      }

      // 设置初始位置和旋转
      if (component.origin) {
        node.position.set(
          component.origin.x,
          component.origin.y,
          component.origin.z
        );
      }

      if (component.rotation && node.userData.initialQuaternion) {
        node.quaternion.copy(node.userData.initialQuaternion);
      }
    });
  }

  /**
   * 添加工具端点可视化
   */
  addToolVisualization() {
    // 创建工具端点的可视化球体
    const toolGeometry = new THREE.SphereGeometry(0.005, 16, 16);
    const toolMesh = new THREE.Mesh(toolGeometry, this.materials.tool);
    toolMesh.name = "tool_sphere";

    // 添加到grasp_base
    if (this.joints["grasp_base"]) {
      toolMesh.position.set(0.00275, 0.0008332, 0.13779);
      this.joints["grasp_base"].add(toolMesh);
    }

    // 存储工具端点引用
    this.toolEndEffector = toolMesh;
  }

  /**
   * 更新关节角度
   * @param {string} jointName - 关节名称
   * @param {number} angle - 角度（弧度）
   */
  updateJoint(jointName, angle) {
    if (!this.isLoaded) return;

    // Prefer exact match (URDF-driven), fallback to lowercase variant for legacy Alicia
    let targetName = jointName;
    if (!this.joints[targetName]) {
      const m = /^Joint(\d+)$/.exec(jointName);
      if (m && this.joints[`joint${m[1]}`]) targetName = `joint${m[1]}`;
    }

    const joint = this.joints[targetName];
    if (!joint) return;

    const component = this.jointHierarchy[targetName] || {
      ...this.extractJointComponent(targetName),
    };
    if (!component || (!component.axis && !joint.userData.axis)) return;

    const initialQuaternion =
      joint.userData.initialQuaternion || new THREE.Quaternion();

    const jointType = joint.userData.jointType || component.type;

    if (jointType === "revolute" || jointType === "continuous") {
      const axis = joint.userData.axis || component.axis;
      const axisVector = new THREE.Vector3(
        axis.x || axis[0],
        axis.y || axis[1],
        axis.z || axis[2]
      );
      const dynamicQuaternion = new THREE.Quaternion().setFromAxisAngle(
        axisVector,
        angle
      );

      // 组合初始旋转和动态旋转
      joint.quaternion.multiplyQuaternions(
        initialQuaternion,
        dynamicQuaternion
      );
    } else if (jointType === "prismatic") {
      const axis = joint.userData.axis || component.axis;
      const axisGroup = joint.userData.axisGroup;
      if (axisGroup) {
        const ax = Array.isArray(axis)
          ? { x: axis[0], y: axis[1], z: axis[2] }
          : axis;
        axisGroup.position.set(angle * ax.x, angle * ax.y, angle * ax.z);
      }
    }
  }

  // 从URDF关节节点推导组件信息（仅在legacy结构缺失时使用）
  extractJointComponent(name) {
    const node = this.joints[name];
    if (!node) return null;
    return {
      type: node.userData.jointType || "fixed",
      axis: node.userData.axis || { x: 0, y: 0, z: 1 },
    };
  }

  /**
   * 批量更新关节角度
   * @param {Object} jointAngles - 关节角度对象
   */
  updateAllJoints(jointAngles) {
    Object.keys(jointAngles).forEach((jointName) => {
      this.updateJoint(jointName, jointAngles[jointName]);
    });

    // 更新末端执行器位置信息
    this.updateEndEffectorPose();
  }

  /**
   * 更新末端执行器位姿信息，Update end effector pose information
   */
  updateEndEffectorPose() {
    if (!this.toolEndEffector) return;

    // 获取工具端点的世界坐标和姿态，Get world position and orientation of tool endpoint
    const worldPosition = new THREE.Vector3();
    const worldQuaternion = new THREE.Quaternion();
    this.toolEndEffector.getWorldPosition(worldPosition);
    this.toolEndEffector.getWorldQuaternion(worldQuaternion);

    // 更新UI显示，Update UI display
    const poseElement = document.getElementById("endEffectorPose");
    if (poseElement) {
      if (window.languageManager) {
        // Use language manager for formatted pose text
        const pose = {
          position: worldPosition,
          orientation: worldQuaternion,
        };
        poseElement.textContent =
          window.languageManager.formatEndEffectorPose(pose);
      } else {
        // Fallback to Chinese
        poseElement.textContent = `x: ${worldPosition.x.toFixed(
          3
        )}, y: ${worldPosition.y.toFixed(3)}, z: ${worldPosition.z.toFixed(
          3
        )} | qx: ${worldQuaternion.x.toFixed(
          3
        )}, qy: ${worldQuaternion.y.toFixed(
          3
        )}, qz: ${worldQuaternion.z.toFixed(
          3
        )}, qw: ${worldQuaternion.w.toFixed(3)}`;
      }
    }
  }

  /**
   * 获取末端执行器位置
   * @returns {THREE.Vector3} 世界坐标位置
   */
  getEndEffectorPosition() {
    if (!this.toolEndEffector) return new THREE.Vector3();

    const worldPosition = new THREE.Vector3();
    this.toolEndEffector.getWorldPosition(worldPosition);
    return worldPosition;
  }

  /**
   * 获取末端执行器方向
   * @returns {THREE.Quaternion} 世界坐标方向
   */
  getEndEffectorOrientation() {
    if (!this.toolEndEffector) return new THREE.Quaternion();

    const worldQuaternion = new THREE.Quaternion();
    this.toolEndEffector.getWorldQuaternion(worldQuaternion);
    return worldQuaternion;
  }

  /**
   * 获取正运动学求解器实例
   * @returns {FKSolver} 正运动学求解器
   */
  getFKSolver() {
    return this.fkSolver;
  }

  /**
   * 高亮显示关节
   * @param {string} jointName - 关节名称
   */
  highlightJoint(jointName) {
    this.clearHighlights();

    const joint = this.joints[jointName];
    if (joint) {
      joint.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.material = this.materials.highlight;
        }
      });
    }
  }

  /**
   * 清除所有高亮
   */
  clearHighlights() {
    Object.keys(this.joints).forEach((jointName) => {
      const joint = this.joints[jointName];
      if (joint) {
        joint.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            // 优先还原原始材质
            if (child.userData && child.userData.originalMaterial) {
              child.material = child.userData.originalMaterial;
            } else if (child.name) {
              child.material = this.getMaterialForPart(child.name);
            } else {
              child.material = this.materials.default;
            }
          }
        });
      }
    });
  }

  /**
   * 设置机械臂可见性
   * @param {boolean} visible - 是否可见
   */
  setVisible(visible) {
    this.robotGroup.visible = visible;
  }

  /**
   * 获取机械臂边界框
   * @returns {THREE.Box3} 边界框
   */
  getBoundingBox() {
    const box = new THREE.Box3();
    box.setFromObject(this.robotGroup);
    return box;
  }

  /**
   * 重置机械臂到初始位置
   */
  resetToHome() {
    const homeAngles = {
      joint1: 0,
      joint2: 0,
      joint3: 0,
      joint4: 0,
      joint5: 0,
      joint6: 0,
      left_finger: 0,
      right_finger: 0,
    };

    this.updateAllJoints(homeAngles);
  }

  /**
   * 获取关节的世界坐标变换
   * @param {string} jointName - 关节名称
   * @returns {THREE.Matrix4} 变换矩阵
   */
  getJointWorldTransform(jointName) {
    const joint = this.joints[jointName];
    if (!joint) return new THREE.Matrix4();

    const matrix = new THREE.Matrix4();
    joint.updateMatrixWorld();
    matrix.copy(joint.matrixWorld);
    return matrix;
  }

  /**
   * 检查是否加载完成
   * @returns {boolean} 是否加载完成
   */
  isLoadingComplete() {
    return this.isLoaded;
  }

  /**
   * 释放资源
   */
  dispose() {
    // 清理坐标系可视化
    this.clearCoordinateFrames();

    // 清理材质
    Object.values(this.materials).forEach((material) => {
      material.dispose();
    });

    // 清理几何体
    Object.values(this.meshes).forEach((mesh) => {
      if (mesh.geometry) {
        mesh.geometry.dispose();
      }
    });

    // 从场景中移除
    if (this.robotGroup.parent) {
      this.robotGroup.parent.remove(this.robotGroup);
    }
  }

  /**
   * 获取URDF中可能的末端候选（叶子link或包含'tool'/'gripper'的link）
   * :return: Array<string>
   */
  getEndEffectorCandidates() {
    // 优先使用URDF图推断：叶子link（不作为任何joint的parent）
    if (this.urdfLinks && this.urdfJointsMap) {
      const parentLinks = new Set();
      Object.keys(this.urdfJointsMap).forEach((jn) => {
        const j = this.urdfJointsMap[jn];
        if (j && j.parent) parentLinks.add(j.parent);
      });
      const candidates = [];
      this.urdfLinks.forEach((linkName) => {
        if (/tool|gripper/i.test(linkName) || !parentLinks.has(linkName)) {
          candidates.push(linkName);
        }
      });
      return Array.from(new Set(candidates));
    }

    // 回退策略：基于场景节点，挑选无子link/joint的link组（忽略Mesh子节点）
    const names = [];
    Object.keys(this.joints).forEach((name) => {
      const node = this.joints[name];
      if (!node || !(node instanceof THREE.Object3D)) return;
      const isJoint = node.userData && node.userData.jointType;
      if (isJoint) return;
      // 统计非Mesh的子节点数量
      const nonMeshChildren = node.children.filter(
        (c) => !(c instanceof THREE.Mesh)
      );
      if (/tool|gripper/i.test(name) || nonMeshChildren.length === 0) {
        names.push(name);
      }
    });
    return Array.from(new Set(names));
  }

  /**
   * 设置末端执行器为指定link（会在其下添加占位点）
   * :param linkName, string
   */
  setEndEffectorByLink(linkName) {
    const node = this.joints[linkName];
    if (!node) return false;
    // 移除旧占位
    if (this.toolEndEffector && this.toolEndEffector.parent) {
      this.toolEndEffector.parent.remove(this.toolEndEffector);
    }
    const placeholder = new THREE.Object3D();
    placeholder.name = "tool_sphere";
    placeholder.position.set(0, 0, 0);
    node.add(placeholder);
    this.toolEndEffector = placeholder;
    // 切换末端后立刻刷新底部位姿显示
    try {
      this.updateEndEffectorPose();
    } catch (e) {}
    return true;
  }

  /**
   * 初始化坐标系可视化
   */
  initializeCoordinateFrames() {
    this.clearCoordinateFrames();

    // 分类：带关节类型信息的节点视为 joint，其余视为 body(link)
    Object.keys(this.joints).forEach((name) => {
      const node = this.joints[name];
      if (!node || !(node instanceof THREE.Object3D)) return;
      const isJoint = !!(node.userData && node.userData.jointType);
      const frame = this._createCoordinateFrameInternal(name);
      if (frame) {
        node.add(frame);
        if (isJoint) {
          this.jointFrames.set(name, frame);
        } else {
          this.bodyFrames.set(name, frame);
        }
      }
    });

    this.updateCoordinateFramesVisibility();
  }

  /**
   * 为指定关节创建坐标系可视化
   * :param jointName, string - 关节名称
   * :param joint, THREE.Object3D - 关节对象
   */
  _createCoordinateFrameInternal(name) {
    const axisLength = 0.08;
    const axisRadius = 0.002;
    const frameGroup = new THREE.Group();
    frameGroup.name = `${name}_coordinate_frame`;
    const axes = [
      { dir: new THREE.Vector3(1, 0, 0), color: 0xff0000 },
      { dir: new THREE.Vector3(0, 1, 0), color: 0x00ff00 },
      { dir: new THREE.Vector3(0, 0, 1), color: 0x0000ff },
    ];
    axes.forEach(a => frameGroup.add(this.createAxisArrow(a.dir, axisLength, axisRadius, a.color)));
    const label = this.createAxisLabel(name);
    label.position.set(0, axisLength + 0.01, 0);
    frameGroup.add(label);
    return frameGroup;
  }

  /**
   * 创建轴箭头
   * :param direction, THREE.Vector3 - 轴方向
   * :param length, number - 轴长度
   * :param radius, number - 轴半径
   * :param color, number - 轴颜色
   * :return: THREE.Group - 轴箭头组
   */
  createAxisArrow(direction, length, radius, color) {
    const group = new THREE.Group();
    
    // 轴杆
    const shaftGeometry = new THREE.CylinderGeometry(radius, radius, length * 0.8, 8);
    const shaftMaterial = new THREE.MeshBasicMaterial({ color: color });
    const shaft = new THREE.Mesh(shaftGeometry, shaftMaterial);
    shaft.position.y = length * 0.4;
    group.add(shaft);
    
    // 箭头头部
    const headGeometry = new THREE.ConeGeometry(radius * 2, length * 0.2, 8);
    const headMaterial = new THREE.MeshBasicMaterial({ color: color });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.y = length * 0.9;
    group.add(head);
    
    // 旋转到正确方向
    const axis = new THREE.Vector3(0, 1, 0);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(axis, direction);
    group.quaternion.copy(quaternion);
    
    return group;
  }

  /**
   * 创建轴标签
   * :param text, string - 标签文本
   * :return: THREE.Sprite - 标签精灵
   */
  createAxisLabel(text) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 128;
    canvas.height = 64;
    
    // 绘制文字
    context.fillStyle = '#ffffff';
    context.font = 'Bold 16px Arial';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, 64, 32);
    
    // 创建纹理和精灵
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: true,
    });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(0.05, 0.025, 1); // 增大标签尺寸
    
    return sprite;
  }

  /**
   * 切换坐标系可视化显示
   * :param show, boolean - 是否显示
   */
  toggleCoordinateFrames(show) { // 兼容旧接口 -> 映射到 mode
    this.setCoordinateFramesMode(show ? 'joint' : 'none');
  }

  /**
   * 设置坐标系显示模式
   * @param {('none'|'body'|'joint')} mode
   */
  setCoordinateFramesMode(mode) {
    if (!['none','body','joint'].includes(mode)) return;
    this.coordinateFramesMode = mode;
    this.updateCoordinateFramesVisibility();
  }

  /**
   * 设置机械臂整体透明/不透明
   * :param {boolean} transparent - 是否透明
   * :param {number} opacity - 透明度 (0-1)
   */
  setTransparency(transparent, opacity = 0.2) {
    const targetOpacity = transparent ? opacity : 1.0;
    const needsTransparent = transparent && targetOpacity < 1.0;
    // 遍历所有mesh
    this.robotGroup.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const mat = child.material;
        if (Array.isArray(mat)) {
            mat.forEach(m => {
              if (!m) return;
              m.transparent = needsTransparent;
              m.opacity = targetOpacity;
              m.depthWrite = !needsTransparent; // 避免半透明排序问题
            });
        } else if (mat) {
          mat.transparent = needsTransparent;
          mat.opacity = targetOpacity;
          mat.depthWrite = !needsTransparent;
        }
      }
    });
  }

  /**
   * 更新坐标系可见性
   */
  updateCoordinateFramesVisibility() {
    const showBody = this.coordinateFramesMode === 'body';
    const showJoint = this.coordinateFramesMode === 'joint';
    this.bodyFrames.forEach(f => { if (f) f.visible = showBody; });
    this.jointFrames.forEach(f => { if (f) f.visible = showJoint; });
  }

  /**
   * 清除所有坐标系可视化
   */
  clearCoordinateFrames() {
    const disposeGroupMap = (map) => {
      map.forEach((frame) => {
        if (frame.parent) frame.parent.remove(frame);
        frame.traverse((child) => {
          if (child.geometry) child.geometry.dispose();
          if (child.material) child.material.dispose();
        });
      });
      map.clear();
    };
    disposeGroupMap(this.bodyFrames);
    disposeGroupMap(this.jointFrames);
    this.coordinateFrames = new Map(); // 保留旧字段但不再使用
  }

  /**
   * 调试方法：强制显示所有坐标系
   */
  debugShowCoordinateFrames() {
    console.log('调试：强制显示所有坐标系');
    if (this.coordinateFrames) {
      this.coordinateFrames.forEach((frame, jointName) => {
        frame.visible = true;
        console.log(`强制显示坐标系: ${jointName}`, frame);
      });
    }
    
    // 也检查场景中的所有对象
    console.log('场景中的所有对象:');
    this.robotGroup.traverse((child) => {
      if (child.name && child.name.includes('coordinate')) {
        console.log('找到坐标系对象:', child.name, child.visible, child.position);
      }
    });
  }

  /**
   * 调试方法：手动切换坐标系显示
   */
  debugToggleCoordinateFrames() {
    console.log('手动切换坐标系显示');
    this.toggleCoordinateFrames(true);
  }
}

// 导出类（如果使用模块系统）
if (typeof module !== "undefined" && module.exports) {
  module.exports = RobotArm;
}
