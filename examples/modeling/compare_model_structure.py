"""Compare model structure between MJCF and URDF formats."""

from __future__ import annotations

import synriard
from synriard import urdf, mjcf
from robocore.modeling.robot_model import RobotModel
from robocore.utils.beauty_logger import beauty_print


def get_all_models():
    """Get all available models from synriard.
    
    :return: List of model dictionaries with name, version, variant, format, and path
    """
    models = []
    
    # Collect models from both URDF and MJCF
    for model_format, model_module in [("urdf", urdf), ("mjcf", mjcf)]:
        # Get all version modules
        version_modules = [attr for attr in dir(model_module) if not attr.startswith('_')]
        
        for version_module_name in version_modules:
            try:
                version_module = getattr(model_module, version_module_name)
                
                # Parse robot name and version from module name
                parts = version_module_name.split('_')
                if len(parts) >= 3:
                    # Try to find version pattern (v followed by numbers)
                    version_idx = None
                    for i in range(1, len(parts)):
                        if parts[i].startswith('v') and parts[i][1:].replace('_', '').isdigit():
                            version_idx = i
                            break
                    
                    if version_idx:
                        name = '_'.join(parts[:version_idx])
                        version = '_'.join(parts[version_idx:])
                    else:
                        name = parts[0]
                        version = '_'.join(parts[1:])
                else:
                    name = parts[0]
                    version = '_'.join(parts[1:]) if len(parts) > 1 else None
                
                # Get all variant objects from version module
                excluded_attrs = {'os', 'SimpleNamespace', 'types', 'abspath', 'dirname', 'join', '__builtins__',
                                  '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_MODULE_PATH'}
                variant_attrs = [attr for attr in dir(version_module)
                                 if not attr.startswith('_') and attr not in excluded_attrs]
                
                for variant_attr in variant_attrs:
                    try:
                        variant_obj = getattr(version_module, variant_attr)
                        # Skip if it's a module or standard library object
                        if isinstance(variant_obj, type) or hasattr(variant_obj, '__module__') and variant_obj.__module__ in ('types', 'os', 'builtins'):
                            continue
                        # Map model format to actual attribute name (mjcf uses 'xml')
                        format_attr = 'xml' if model_format == 'mjcf' else model_format
                        # Check if this variant object has the requested format
                        if hasattr(variant_obj, format_attr):
                            path = getattr(variant_obj, format_attr)
                            
                            # Extract variant name
                            if name == "Alicia_D":
                                pattern = f"{name}_{version}_"
                                if variant_attr.startswith(pattern):
                                    variant = variant_attr.replace(pattern, "")
                                else:
                                    variant = None
                            elif name == "Alicia_M":
                                pattern = f"{name}_{version}_gripper_"
                                if variant_attr.startswith(pattern):
                                    variant = variant_attr.replace(pattern, "")
                                    variant = f"gripper_{variant}"
                                else:
                                    variant = None
                            elif name == "Bessica_D":
                                pattern = f"{name}_{version}_"
                                if variant_attr.startswith(pattern):
                                    variant = variant_attr.replace(pattern, "")
                                else:
                                    variant = None
                            elif name == "Bessica_M":
                                variant = None
                            else:
                                variant = None
                            
                            models.append({
                                'name': name,
                                'version': version,
                                'variant': variant,
                                'format': model_format,
                                'path': path
                            })
                    except (AttributeError, TypeError):
                        continue
                        
            except (AttributeError, TypeError):
                continue
    
    return models


def group_models_by_identity(models):
    """Group models by name, version, and variant.
    
    :param models: List of model dictionaries
    :return: Dictionary mapping (name, version, variant) to list of (format, path) tuples
    """
    grouped = {}
    for model in models:
        key = (model['name'], model['version'], model['variant'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][model['format']] = model['path']
    return grouped


def extract_structure(model):
    """Extract complete link-joint structure from a robot model.
    
    :param model: RobotModel instance
    :return: Dictionary mapping parent_link -> list of (joint_name, joint_type, child_link) tuples
    """
    structure = {}
    visited_links = set()
    
    def traverse(link):
        if link in visited_links:
            return
        visited_links.add(link)
        
        children = model._graph.get(link, [])
        structure[link] = []
        for joint in children:
            structure[link].append((joint.name, joint.joint_type, joint.child))
            traverse(joint.child)
    
    traverse(model.base_link)
    return structure


def compare_structures(mjcf_structure, urdf_structure, mjcf_base, urdf_base):
    """Compare two model structures and find differences.
    
    :param mjcf_structure: Structure dict from MJCF model
    :param urdf_structure: Structure dict from URDF model
    :param mjcf_base: Base link name from MJCF
    :param urdf_base: Base link name from URDF
    :return: Dictionary with differences
    """
    differences = {
        'base_link_diff': mjcf_base != urdf_base,
        'missing_links_mjcf': [],
        'missing_links_urdf': [],
        'extra_links_mjcf': [],
        'extra_links_urdf': [],
        'link_children_diff': {},
        'joint_name_diff': {},
        'joint_type_diff': {}
    }
    
    # Collect all links from both structures
    all_mjcf_links = set(mjcf_structure.keys())
    all_urdf_links = set(urdf_structure.keys())
    
    # Add child links
    for children in mjcf_structure.values():
        for _, _, child in children:
            all_mjcf_links.add(child)
    for children in urdf_structure.values():
        for _, _, child in children:
            all_urdf_links.add(child)
    
    # Find missing links
    differences['missing_links_mjcf'] = sorted(all_urdf_links - all_mjcf_links)
    differences['missing_links_urdf'] = sorted(all_mjcf_links - all_urdf_links)
    
    # Find common links and compare their children
    common_links = all_mjcf_links & all_urdf_links
    
    for link in common_links:
        mjcf_children = mjcf_structure.get(link, [])
        urdf_children = urdf_structure.get(link, [])
        
        # Create maps: child_link -> (joint_name, joint_type)
        mjcf_child_map = {child: (jname, jtype) for jname, jtype, child in mjcf_children}
        urdf_child_map = {child: (jname, jtype) for jname, jtype, child in urdf_children}
        
        mjcf_child_links = set(mjcf_child_map.keys())
        urdf_child_links = set(urdf_child_map.keys())
        
        if mjcf_child_links != urdf_child_links:
            differences['link_children_diff'][link] = {
                'mjcf_children': sorted(mjcf_child_links),
                'urdf_children': sorted(urdf_child_links),
                'missing_in_mjcf': sorted(urdf_child_links - mjcf_child_links),
                'missing_in_urdf': sorted(mjcf_child_links - urdf_child_links)
            }
        
        # Compare joint names and types for common child links
        common_children = mjcf_child_links & urdf_child_links
        for child in common_children:
            mjcf_jname, mjcf_jtype = mjcf_child_map[child]
            urdf_jname, urdf_jtype = urdf_child_map[child]
            
            if mjcf_jname != urdf_jname:
                if link not in differences['joint_name_diff']:
                    differences['joint_name_diff'][link] = []
                differences['joint_name_diff'][link].append({
                    'child': child,
                    'mjcf': mjcf_jname,
                    'urdf': urdf_jname
                })
            
            if mjcf_jtype != urdf_jtype:
                if link not in differences['joint_type_diff']:
                    differences['joint_type_diff'][link] = []
                differences['joint_type_diff'][link].append({
                    'child': child,
                    'mjcf': mjcf_jtype,
                    'urdf': urdf_jtype
                })
    
    return differences


def print_differences(name, version, variant, differences):
    """Print structural differences between MJCF and URDF.
    
    :param name: Robot name
    :param version: Robot version
    :param variant: Robot variant
    :param differences: Dictionary of differences from compare_structures
    :return: True if differences found, False otherwise
    """
    model_id = f"{name}"
    if version:
        model_id += f" {version}"
    if variant:
        model_id += f" ({variant})"
    
    has_differences = False
    
    # Base link difference
    if differences['base_link_diff']:
        beauty_print(f"⚠️  Base link differs: MJCF={differences.get('mjcf_base', 'N/A')}, URDF={differences.get('urdf_base', 'N/A')}", type="warning")
        has_differences = True
    
    # Missing links
    if differences['missing_links_mjcf']:
        if len(differences['missing_links_mjcf']) <= 5:
            beauty_print(f"⚠️  Links in URDF but missing in MJCF ({len(differences['missing_links_mjcf'])}): {', '.join(differences['missing_links_mjcf'])}", type="warning")
        else:
            beauty_print(f"⚠️  Links in URDF but missing in MJCF ({len(differences['missing_links_mjcf'])}): {', '.join(differences['missing_links_mjcf'][:5])} ... (and {len(differences['missing_links_mjcf'])-5} more)", type="warning")
        has_differences = True
    
    if differences['missing_links_urdf']:
        if len(differences['missing_links_urdf']) <= 5:
            beauty_print(f"⚠️  Links in MJCF but missing in URDF ({len(differences['missing_links_urdf'])}): {', '.join(differences['missing_links_urdf'])}", type="warning")
        else:
            beauty_print(f"⚠️  Links in MJCF but missing in URDF ({len(differences['missing_links_urdf'])}): {', '.join(differences['missing_links_urdf'][:5])} ... (and {len(differences['missing_links_urdf'])-5} more)", type="warning")
        has_differences = True
    
    # Link children differences
    if differences['link_children_diff']:
        beauty_print(f"⚠️  Link children structure differences ({len(differences['link_children_diff'])} links):", type="warning")
        for link, diff_info in differences['link_children_diff'].items():
            if diff_info['missing_in_mjcf']:
                beauty_print(f"   - Link '{link}': URDF has children missing in MJCF: {', '.join(diff_info['missing_in_mjcf'])}", type="warning")
            if diff_info['missing_in_urdf']:
                beauty_print(f"   - Link '{link}': MJCF has children missing in URDF: {', '.join(diff_info['missing_in_urdf'])}", type="warning")
        has_differences = True
    
    # Joint name differences
    if differences['joint_name_diff']:
        beauty_print(f"⚠️  Joint name differences ({len(differences['joint_name_diff'])} links):", type="warning")
        for link, name_diffs in differences['joint_name_diff'].items():
            for diff in name_diffs:
                beauty_print(f"   - Link '{link}' -> '{diff['child']}': MJCF='{diff['mjcf']}', URDF='{diff['urdf']}'", type="warning")
        has_differences = True
    
    # Joint type differences
    if differences['joint_type_diff']:
        beauty_print(f"⚠️  Joint type differences ({len(differences['joint_type_diff'])} links):", type="warning")
        for link, type_diffs in differences['joint_type_diff'].items():
            for diff in type_diffs:
                beauty_print(f"   - Link '{link}' -> '{diff['child']}': MJCF='{diff['mjcf']}', URDF='{diff['urdf']}'", type="warning")
        has_differences = True
    
    return has_differences


def compare_model_structures(name, version, variant, mjcf_path=None, urdf_path=None):
    """Compare structure between MJCF and URDF formats.
    
    :param name: Robot name
    :param version: Robot version
    :param variant: Robot variant
    :param mjcf_path: Path to MJCF file (optional)
    :param urdf_path: Path to URDF file (optional)
    """
    if not mjcf_path or not urdf_path:
        return
    
    try:
        model_id = f"{name}"
        if version:
            model_id += f" {version}"
        if variant:
            model_id += f" ({variant})"
        
        beauty_print(f"Model: {model_id}", type="module", centered=True)
        
        mjcf_model = RobotModel(mjcf_path)
        urdf_model = RobotModel(urdf_path)
        
        mjcf_structure = extract_structure(mjcf_model)
        urdf_structure = extract_structure(urdf_model)
        
        differences = compare_structures(mjcf_structure, urdf_structure, mjcf_model.base_link, urdf_model.base_link)
        differences['mjcf_base'] = mjcf_model.base_link
        differences['urdf_base'] = urdf_model.base_link
        
        # Check if there are any differences
        has_differences = (
            differences['base_link_diff'] or
            differences['missing_links_mjcf'] or
            differences['missing_links_urdf'] or
            differences['link_children_diff'] or
            differences['joint_name_diff'] or
            differences['joint_type_diff']
        )
        
        if has_differences:
            # Print differences summary first
            print_differences(name, version, variant, differences)
            
            # Print full structure trees for both formats
            beauty_print(f"[MJCF] Full structure tree for {model_id}", type="info")
            mjcf_model.print_tree(show_joints=True, show_fixed=False)
            print()
            
            beauty_print(f"[URDF] Full structure tree for {model_id}", type="info")
            urdf_model.print_tree(show_joints=True, show_fixed=False)
            print()
        else:
            beauty_print(f"✓ No structural differences found between MJCF and URDF", type="success")
            print()
        
    except Exception as e:
        beauty_print(f"Failed to compare structures: {e}", type="error")
        print()


def print_model_structure(name, version, variant, mjcf_path=None, urdf_path=None):
    """Print structure for a model in both MJCF and URDF formats.
    
    :param name: Robot name
    :param version: Robot version
    :param variant: Robot variant
    :param mjcf_path: Path to MJCF file (optional)
    :param urdf_path: Path to URDF file (optional)
    """
    model_id = f"{name}"
    if version:
        model_id += f" {version}"
    if variant:
        model_id += f" ({variant})"
    
    beauty_print(f"Model: {model_id}", type="module", centered=True)
    
    # Print MJCF structure first
    if mjcf_path:
        try:
            beauty_print(f"[MJCF] Structure for {model_id}", type="info")
            mjcf_model = RobotModel(mjcf_path)
            mjcf_model.print_tree(show_joints=True, show_fixed=False)
            print()  # Empty line for spacing
        except Exception as e:
            beauty_print(f"Failed to load MJCF for {model_id}: {e}", type="error")
            print()
    else:
        beauty_print(f"[MJCF] Not available for {model_id}", type="warning")
        print()
    
    # Print URDF structure
    if urdf_path:
        try:
            beauty_print(f"[URDF] Structure for {model_id}", type="info")
            urdf_model = RobotModel(urdf_path)
            urdf_model.print_tree(show_joints=True, show_fixed=False)
            print()  # Empty line for spacing
        except Exception as e:
            beauty_print(f"Failed to load URDF for {model_id}: {e}", type="warning")
            print()
    else:
        beauty_print(f"[URDF] Not available for {model_id}", type="warning")
        print()


def main():
    """Main function to compare structures between MJCF and URDF formats."""
    beauty_print("Collecting all available models from synriard...", type="info")
    all_models = get_all_models()
    
    # Group models by identity (name, version, variant)
    grouped = group_models_by_identity(all_models)
    
    beauty_print(f"Found {len(grouped)} unique model configurations", type="success")
    print()
    
    # Iterate through each unique model configuration
    for (name, version, variant), format_paths in sorted(grouped.items()):
        # Get paths for both formats
        mjcf_path = format_paths.get('mjcf')
        urdf_path = format_paths.get('urdf')
        
        # Compare structures between MJCF and URDF
        compare_model_structures(name, version, variant, mjcf_path, urdf_path)


if __name__ == "__main__":
    main()
