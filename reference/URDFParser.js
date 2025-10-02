/**
 * URDF解析器类
 * 用于解析URDF文件并提取机械臂的结构信息
 */
class URDFParser {
  constructor() {
    this.parser = new DOMParser();
  }

  /**
   * 解析URDF文件
   * @param {string} urdfContent, URDF文件内容
   * @param {string} baseDir, 基础目录路径（用于解析mesh文件路径）
   * @returns {Object} 解析后的机械臂结构信息
   */
  parseURDF(urdfContent, baseDir = '') {
    try {
      const xmlDoc = this.parser.parseFromString(urdfContent, 'text/xml');
      const robot = xmlDoc.querySelector('robot');
      
      if (!robot) {
        throw new Error('Invalid URDF file: No robot element found');
      }

      const robotName = robot.getAttribute('name');
      
      // 解析顶层材质库（供visual内引用）
      const materials = this.parseMaterials(xmlDoc, baseDir);

      // 解析所有链接
      const links = this.parseLinks(xmlDoc, baseDir, materials);
      
      // 解析所有关节
      const joints = this.parseJoints(xmlDoc);
      
      // 构建层次结构
      const hierarchy = this.buildHierarchy(links, joints);
      
      // 获取所有mesh文件列表
      const meshFiles = this.extractMeshFiles(links);

      return {
        name: robotName,
        links: links,
        joints: joints,
        hierarchy: hierarchy,
        meshFiles: meshFiles,
        baseDir: baseDir,
        materials: materials
      };
    } catch (error) {
      console.error('URDF parsing error:', error);
      throw error;
    }
  }

  /**
   * 解析链接元素
   * @param {Document} xmlDoc, XML文档
   * @param {string} baseDir, 基础目录
   * @returns {Object} 链接信息对象
   */
  parseLinks(xmlDoc, baseDir, materialsMap) {
    const links = {};
    const linkElements = xmlDoc.querySelectorAll('link');
    
    linkElements.forEach(linkEl => {
      const name = linkEl.getAttribute('name');
      const link = {
        name: name,
        visual: null,
        collision: null,
        inertial: null
      };

      // 解析visual元素
      const visual = linkEl.querySelector('visual');
      if (visual) {
        link.visual = this.parseVisual(visual, baseDir, materialsMap);
      }

      // 解析collision元素
      const collision = linkEl.querySelector('collision');
      if (collision) {
        link.collision = this.parseCollision(collision, baseDir);
      }

      // 解析inertial元素
      const inertial = linkEl.querySelector('inertial');
      if (inertial) {
        link.inertial = this.parseInertial(inertial);
      }

      links[name] = link;
    });

    return links;
  }

  /**
   * 解析visual元素
   * @param {Element} visual, visual元素
   * @param {string} baseDir, 基础目录
   * @returns {Object} visual信息
   */
  parseVisual(visual, baseDir, materialsMap) {
    const origin = this.parseOrigin(visual.querySelector('origin'));
    const geometry = this.parseGeometry(visual.querySelector('geometry'), baseDir);
    const material = this.parseMaterial(visual.querySelector('material'), baseDir, materialsMap);

    return { origin, geometry, material };
  }

  /**
   * 解析collision元素
   * @param {Element} collision, collision元素
   * @param {string} baseDir, 基础目录
   * @returns {Object} collision信息
   */
  parseCollision(collision, baseDir) {
    const origin = this.parseOrigin(collision.querySelector('origin'));
    const geometry = this.parseGeometry(collision.querySelector('geometry'), baseDir);

    return { origin, geometry };
  }

  /**
   * 解析inertial元素
   * @param {Element} inertial, inertial元素
   * @returns {Object} inertial信息
   */
  parseInertial(inertial) {
    const origin = this.parseOrigin(inertial.querySelector('origin'));
    const mass = parseFloat(inertial.querySelector('mass')?.getAttribute('value') || '0');
    
    const inertiaEl = inertial.querySelector('inertia');
    const inertiaData = {};
    if (inertiaEl) {
      ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'].forEach(prop => {
        inertiaData[prop] = parseFloat(inertiaEl.getAttribute(prop) || '0');
      });
    }

    return { origin, mass, inertia: inertiaData };
  }

  /**
   * 解析origin元素
   * @param {Element} origin, origin元素
   * @returns {Object} origin信息
   */
  parseOrigin(origin) {
    if (!origin) {
      return { xyz: [0, 0, 0], rpy: [0, 0, 0] };
    }

    const xyz = (origin.getAttribute('xyz') || '0 0 0').split(' ').map(parseFloat);
    const rpy = (origin.getAttribute('rpy') || '0 0 0').split(' ').map(parseFloat);

    return { xyz, rpy };
  }

  /**
   * 解析geometry元素
   * @param {Element} geometry, geometry元素
   * @param {string} baseDir, 基础目录
   * @returns {Object} geometry信息
   */
  parseGeometry(geometry, baseDir) {
    if (!geometry) return null;

    const mesh = geometry.querySelector('mesh');
    if (mesh) {
      let filename = mesh.getAttribute('filename');
      if (filename) {
        // 支持相对路径：相对于baseDir（可以是 http(s):// 或 file:// 或相对URL）
        const hasScheme = /^[a-zA-Z]+:\/\//.test(filename);
        if (filename.startsWith("package://")) {
          const rel = filename.replace("package://", "");
          filename = baseDir ? `${baseDir}/${rel}` : rel;
        } else if (!hasScheme) {
          // 去掉开头的'./'
          filename = filename.replace(/^\.\//, "");
          if (baseDir) {
            // 规范化baseDir尾部斜杠
            const bd = baseDir.replace(/\/$/, "");
            filename = `${bd}/${filename}`;
          }
        }
      }
      
      const scale = mesh.getAttribute('scale');
      return {
        type: 'mesh',
        filename: filename,
        scale: scale ? scale.split(' ').map(parseFloat) : [1, 1, 1]
      };
    }

    // 其他几何形状类型
    const box = geometry.querySelector('box');
    if (box) {
      const size = box.getAttribute('size').split(' ').map(parseFloat);
      return { type: 'box', size };
    }

    const cylinder = geometry.querySelector('cylinder');
    if (cylinder) {
      return {
        type: 'cylinder',
        radius: parseFloat(cylinder.getAttribute('radius')),
        length: parseFloat(cylinder.getAttribute('length'))
      };
    }

    const sphere = geometry.querySelector('sphere');
    if (sphere) {
      return {
        type: 'sphere',
        radius: parseFloat(sphere.getAttribute('radius'))
      };
    }

    return null;
  }

  /**
   * 解析material元素
   * @param {Element} material, material元素
   * @returns {Object} material信息
   */
  parseMaterial(material, baseDir, materialsMap) {
    if (!material) return null;

    // Inline definition present
    const colorEl = material.querySelector("color");
    const textureEl = material.querySelector("texture");
    const nameAttr = material.getAttribute("name") || undefined;

    if (colorEl || textureEl) {
      const rgba = colorEl
        ? colorEl.getAttribute("rgba").split(" ").map(parseFloat)
        : [0.8, 0.8, 0.8, 1.0];

      let texturePath = textureEl ? textureEl.getAttribute("filename") : null;
      if (texturePath) {
        // Normalize texture path with baseDir and package:// support
        const hasScheme = /^[a-zA-Z]+:\/\//.test(texturePath);
        if (texturePath.startsWith("package://")) {
          const rel = texturePath.replace("package://", "");
          texturePath = baseDir ? `${baseDir}/${rel}` : rel;
        } else if (!hasScheme) {
          texturePath = texturePath.replace(/^\.\//, "");
          if (baseDir) {
            const bd = baseDir.replace(/\/$/, "");
            texturePath = `${bd}/${texturePath}`;
          }
        }
      }

      return {
        name: nameAttr,
        color: rgba,
        texture: texturePath || null,
      };
    }

    // Reference by name only -> resolve from materials map
    const refName = material.getAttribute("name");
    if (refName && materialsMap && materialsMap[refName]) {
      return {
        name: refName,
        color: Array.isArray(materialsMap[refName].color)
          ? materialsMap[refName].color.slice()
          : [0.8, 0.8, 0.8, 1.0],
        texture: materialsMap[refName].texture || null,
      };
    }

    // Fallback default
    return {
      name: refName || undefined,
      color: [0.8, 0.8, 0.8, 1.0],
      texture: null,
    };
  }

  /**
   * Parse top-level materials under <robot> for later reference
   *
   * :param xmlDoc, Document
   * :param baseDir, string
   * :return: Object{name -> {color:Array[4], texture:string|null}}
   */
  parseMaterials(xmlDoc, baseDir) {
    const map = {};
    const robot = xmlDoc.querySelector('robot');
    if (!robot) return map;

    // Only direct children of <robot>
    let materialEls = [];
    try {
      materialEls = robot.querySelectorAll(':scope > material');
    } catch (e) {
      // Fallback for environments without :scope support
      materialEls = xmlDoc.querySelectorAll('robot > material');
    }
    materialEls.forEach((matEl) => {
      const name = matEl.getAttribute('name');
      if (!name) return;
      // Do not overwrite existing entries (first definition wins)
      if (Object.prototype.hasOwnProperty.call(map, name)) return;

      const colorEl = matEl.querySelector('color');
      const textureEl = matEl.querySelector('texture');
      const rgba = colorEl
        ? colorEl.getAttribute('rgba').split(' ').map(parseFloat)
        : undefined;

      let texturePath = textureEl ? textureEl.getAttribute('filename') : null;
      if (texturePath) {
        const hasScheme = /^[a-zA-Z]+:\/\//.test(texturePath);
        if (texturePath.startsWith('package://')) {
          const rel = texturePath.replace('package://', '');
          texturePath = baseDir ? `${baseDir}/${rel}` : rel;
        } else if (!hasScheme) {
          texturePath = texturePath.replace(/^\.\//, '');
          if (baseDir) {
            const bd = baseDir.replace(/\/$/, '');
            texturePath = `${bd}/${texturePath}`;
          }
        }
      }

      map[name] = {
        color: rgba || [0.8, 0.8, 0.8, 1.0],
        texture: texturePath || null,
      };
    });

    return map;
  }

  /**
   * 解析关节元素
   * @param {Document} xmlDoc, XML文档
   * @returns {Object} 关节信息对象
   */
  parseJoints(xmlDoc) {
    const joints = {};
    const jointElements = xmlDoc.querySelectorAll('joint');
    
    jointElements.forEach(jointEl => {
      const name = jointEl.getAttribute('name');
      const type = jointEl.getAttribute('type');
      
      const parent = jointEl.querySelector('parent')?.getAttribute('link');
      const child = jointEl.querySelector('child')?.getAttribute('link');
      const origin = this.parseOrigin(jointEl.querySelector('origin'));
      
      const axisEl = jointEl.querySelector('axis');
      const axis = axisEl ? 
        axisEl.getAttribute('xyz').split(' ').map(parseFloat) : [0, 0, 1];

      const limitEl = jointEl.querySelector('limit');
      const limit = {};
      if (limitEl) {
        limit.lower = parseFloat(limitEl.getAttribute('lower') || '0');
        limit.upper = parseFloat(limitEl.getAttribute('upper') || '0');
        limit.effort = parseFloat(limitEl.getAttribute('effort') || '0');
        limit.velocity = parseFloat(limitEl.getAttribute('velocity') || '0');
      }

      joints[name] = {
        name: name,
        type: type,
        parent: parent,
        child: child,
        origin: origin,
        axis: axis,
        limit: limit
      };
    });

    return joints;
  }

  /**
   * 构建层次结构
   * @param {Object} links, 链接信息
   * @param {Object} joints, 关节信息
   * @returns {Object} 层次结构信息
   */
  buildHierarchy(links, joints) {
    const hierarchy = {};
    
    // 初始化所有链接
    Object.keys(links).forEach(linkName => {
      hierarchy[linkName] = {
        ...links[linkName],
        children: [],
        jointToChild: null
      };
    });

    // 添加关节到层次结构
    Object.keys(joints).forEach(jointName => {
      const joint = joints[jointName];
      hierarchy[jointName] = {
        ...joint,
        children: []
      };
    });

    // 建立父子关系
    Object.values(joints).forEach(joint => {
      const parentLink = hierarchy[joint.parent];
      const childLink = hierarchy[joint.child];
      const jointNode = hierarchy[joint.name];
      
      if (parentLink && jointNode && childLink) {
        // 链接 -> 关节 -> 链接
        parentLink.children.push(joint.name);
        parentLink.jointToChild = joint.name;
        jointNode.children.push(joint.child);
      }
    });

    return hierarchy;
  }

  /**
   * 提取所有mesh文件
   * @param {Object} links, 链接信息
   * @returns {Array} mesh文件列表
   */
  extractMeshFiles(links) {
    const meshFiles = [];
    
    Object.values(links).forEach(link => {
      if (link.visual && link.visual.geometry && link.visual.geometry.type === 'mesh') {
        const filename = link.visual.geometry.filename;
        if (filename && !meshFiles.includes(filename)) {
          meshFiles.push(filename);
        }
      }
      
      if (link.collision && link.collision.geometry && link.collision.geometry.type === 'mesh') {
        const filename = link.collision.geometry.filename;
        if (filename && !meshFiles.includes(filename)) {
          meshFiles.push(filename);
        }
      }
    });

    return meshFiles;
  }

  /**
   * 查找根链接（没有父关节的链接）
   * @param {Object} joints, 关节信息
   * @param {Object} links, 链接信息
   * @returns {string} 根链接名称
   */
  findRootLink(joints, links) {
    const childLinks = new Set();
    Object.values(joints).forEach(joint => {
      childLinks.add(joint.child);
    });

    const rootLinks = Object.keys(links).filter(linkName => !childLinks.has(linkName));
    return rootLinks.length > 0 ? rootLinks[0] : Object.keys(links)[0];
  }
}

// 导出类（如果使用模块系统）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = URDFParser;
} 