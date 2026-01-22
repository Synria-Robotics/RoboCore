"""Compare forward kinematics between MJCF and URDF models."""

from __future__ import annotations

import numpy as np
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
    :return: Dictionary mapping (name, version, variant) to dict of format -> path
    """
    grouped = {}
    for model in models:
        key = (model['name'], model['version'], model['variant'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][model['format']] = model['path']
    return grouped


def find_common_links(mjcf_model, urdf_model):
    """Find common links between two models.
    
    :param mjcf_model: MJCF RobotModel instance
    :param urdf_model: URDF RobotModel instance
    :return: List of common link names
    """
    mjcf_links = set(mjcf_model.real_link)
    urdf_links = set(urdf_model.real_link)
    return sorted(list(mjcf_links & urdf_links))


def find_common_end_effectors(mjcf_model, urdf_model):
    """Find common end effectors or suitable comparison links.
    
    :param mjcf_model: MJCF RobotModel instance
    :param urdf_model: URDF RobotModel instance
    :return: List of tuples (mjcf_link, urdf_link) for comparison
    """
    common_links = find_common_links(mjcf_model, urdf_model)
    
    # If end links are the same, use them
    if mjcf_model.end_link == urdf_model.end_link and mjcf_model.end_link in common_links:
        return [(mjcf_model.end_link, urdf_model.end_link)]
    
    # Try to find matching end effectors (e.g., left_tool0 vs right_tool0)
    mjcf_end = mjcf_model.end_link
    urdf_end = urdf_model.end_link
    
    # Check if we can find a matching link (e.g., left_tool0 <-> right_tool0)
    if 'left' in mjcf_end and 'right' in urdf_end:
        # Try swapping left/right
        swapped_mjcf = mjcf_end.replace('left', 'right')
        swapped_urdf = urdf_end.replace('right', 'left')
        if swapped_mjcf in common_links and swapped_urdf in common_links:
            return [(swapped_mjcf, swapped_urdf)]
        if mjcf_end.replace('left', 'right') == urdf_end or urdf_end.replace('right', 'left') == mjcf_end:
            # They are symmetric, compare each with its own end
            return [(mjcf_end, urdf_end)]
    
    # Find all leaf links (potential end effectors)
    # Build set of links with children
    def get_leaf_links(model):
        links_with_children = set()
        for link_name in model._graph:
            if link_name in model.real_link:
                for joint in model._graph[link_name]:
                    if joint.child in model.real_link:
                        links_with_children.add(link_name)
        leaf_links = []
        for link_name in model.real_link:
            if link_name != model.base_link and link_name not in links_with_children:
                leaf_links.append(link_name)
        return leaf_links
    
    mjcf_leaves = get_leaf_links(mjcf_model)
    urdf_leaves = get_leaf_links(urdf_model)
    
    # Try to match leaf links
    comparisons = []
    for mjcf_leaf in mjcf_leaves:
        if mjcf_leaf in common_links:
            # Try exact match first
            if mjcf_leaf in urdf_leaves:
                comparisons.append((mjcf_leaf, mjcf_leaf))
            else:
                # Try to find a similar link in URDF
                for urdf_leaf in urdf_leaves:
                    if urdf_leaf in common_links:
                        # Check if they're symmetric variants
                        if (('left' in mjcf_leaf and 'right' in urdf_leaf) or 
                            ('right' in mjcf_leaf and 'left' in urdf_leaf)):
                            if mjcf_leaf.replace('left', 'right').replace('right', 'left') == urdf_leaf:
                                comparisons.append((mjcf_leaf, urdf_leaf))
    
    if comparisons:
        return comparisons
    
    # Fallback: compare default end links even if different
    if mjcf_end in common_links and urdf_end in common_links:
        return [(mjcf_end, urdf_end)]
    
    # Last resort: compare any common link that's furthest from base
    if common_links:
        # Use the last common link (likely furthest from base)
        last_common = common_links[-1]
        return [(last_common, last_common)]
    
    return []


def compare_fk(mjcf_path, urdf_path, num_test_cases=10, tolerance=1e-5):
    """Compare forward kinematics between MJCF and URDF models.
    
    :param mjcf_path: Path to MJCF file
    :param urdf_path: Path to URDF file
    :param num_test_cases: Number of random test configurations
    :param tolerance: Tolerance for comparing poses
    :return: Dictionary with comparison results
    """
    try:
        mjcf_model = RobotModel(mjcf_path)
        urdf_model = RobotModel(urdf_path)
    except Exception as e:
        return {
            'error': str(e), 
            'success': False,
            'mjcf_path': mjcf_path,
            'urdf_path': urdf_path
        }
    
    # Find comparison links
    comparison_links = find_common_end_effectors(mjcf_model, urdf_model)
    
    if not comparison_links:
        return {
            'error': f"No common links found for comparison. MJCF links: {mjcf_model.real_link[:5]}..., URDF links: {urdf_model.real_link[:5]}...",
            'success': False,
            'mjcf_path': mjcf_path,
            'urdf_path': urdf_path,
            'mjcf_model': mjcf_model,
            'urdf_model': urdf_model
        }
    
    mjcf_link, urdf_link = comparison_links[0]
    
    # Check DOF - if different, we'll still try to compare but note the difference
    dof_match = mjcf_model.num_chain_dof == urdf_model.num_chain_dof
    if not dof_match:
        # Try to find a common chain
        # For now, we'll compare using the smaller DOF
        min_dof = min(mjcf_model.num_chain_dof, urdf_model.num_chain_dof)
        if min_dof == 0:
            return {
                'error': f"DOF mismatch: MJCF={mjcf_model.num_chain_dof}, URDF={urdf_model.num_chain_dof}",
                'success': False,
                'mjcf_end_link': mjcf_model.end_link,
                'urdf_end_link': urdf_model.end_link,
                'mjcf_path': mjcf_path,
                'urdf_path': urdf_path,
                'mjcf_model': mjcf_model,
                'urdf_model': urdf_model
            }
    else:
        min_dof = mjcf_model.num_chain_dof
    
    results = {
        'success': True,
        'num_dof_mjcf': mjcf_model.num_chain_dof,
        'num_dof_urdf': urdf_model.num_chain_dof,
        'dof_match': dof_match,
        'mjcf_end_link': mjcf_model.end_link,
        'urdf_end_link': urdf_model.end_link,
        'comparison_link_mjcf': mjcf_link,
        'comparison_link_urdf': urdf_link,
        'test_cases': [],
        'max_position_error': 0.0,
        'max_rotation_error': 0.0,
        'all_match': True,
        'mjcf_path': mjcf_path,
        'urdf_path': urdf_path,
        'mjcf_model': mjcf_model,
        'urdf_model': urdf_model
    }
    
    # Generate random test configurations
    for i in range(num_test_cases):
        # Use random_q_batch to generate valid joint configurations
        # Use the model with non-zero DOF
        try:
            if mjcf_model.num_chain_dof > 0:
                q_mjcf = mjcf_model.random_q_batch(1, seed=i)[0]
            else:
                q_mjcf = np.array([])
            
            if urdf_model.num_chain_dof > 0:
                q_urdf = urdf_model.random_q_batch(1, seed=i)[0]
            else:
                q_urdf = np.array([])
        except:
            # Fallback: generate random angles within limits
            if mjcf_model.num_chain_dof > 0:
                q_mjcf = np.random.uniform(
                    mjcf_model.joint_limits_min[:mjcf_model.num_chain_dof],
                    mjcf_model.joint_limits_max[:mjcf_model.num_chain_dof]
                )
            else:
                q_mjcf = np.array([])
            
            if urdf_model.num_chain_dof > 0:
                q_urdf = np.random.uniform(
                    urdf_model.joint_limits_min[:urdf_model.num_chain_dof],
                    urdf_model.joint_limits_max[:urdf_model.num_chain_dof]
                )
            else:
                q_urdf = np.array([])
        
        # Compute FK for both models at the comparison links
        try:
            # If DOF match, use same q for both
            if dof_match and len(q_mjcf) > 0:
                q = q_mjcf
                mjcf_pose = mjcf_model.fk(q, return_end=False, end_link=mjcf_link)
                urdf_pose = urdf_model.fk(q, return_end=False, end_link=urdf_link)
            else:
                # Different DOF - compare at the specified links
                if mjcf_model.num_chain_dof > 0:
                    mjcf_pose = mjcf_model.fk(q_mjcf, return_end=False, end_link=mjcf_link)
                else:
                    # Zero DOF - get base pose
                    mjcf_pose = mjcf_model.fk([], return_end=False, end_link=mjcf_link)
                
                if urdf_model.num_chain_dof > 0:
                    urdf_pose = urdf_model.fk(q_urdf, return_end=False, end_link=urdf_link)
                else:
                    urdf_pose = urdf_model.fk([], return_end=False, end_link=urdf_link)
            
            # Extract pose for the specific link
            if isinstance(mjcf_pose, dict):
                mjcf_pose = mjcf_pose.get(mjcf_link, mjcf_pose.get(list(mjcf_pose.keys())[0]))
            if isinstance(urdf_pose, dict):
                urdf_pose = urdf_pose.get(urdf_link, urdf_pose.get(list(urdf_pose.keys())[0]))
                
        except Exception as e:
            results['test_cases'].append({
                'index': i,
                'error': str(e),
                'match': False
            })
            results['all_match'] = False
            continue
        
        # Convert to numpy if needed
        if hasattr(mjcf_pose, 'cpu'):
            mjcf_pose = mjcf_pose.cpu().numpy()
        if hasattr(urdf_pose, 'cpu'):
            urdf_pose = urdf_pose.cpu().numpy()
        
        mjcf_pose = np.array(mjcf_pose)
        urdf_pose = np.array(urdf_pose)
        
        # Compare poses
        # Position error
        mjcf_pos = mjcf_pose[:3, 3]
        urdf_pos = urdf_pose[:3, 3]
        position_error = np.linalg.norm(mjcf_pos - urdf_pos)
        
        # Rotation error (angle between rotation matrices)
        mjcf_rot = mjcf_pose[:3, :3]
        urdf_rot = urdf_pose[:3, :3]
        rot_diff = mjcf_rot @ urdf_rot.T
        # Compute angle from rotation matrix
        trace = np.trace(rot_diff)
        trace = np.clip(trace, -1.0, 3.0)
        # Clamp to avoid numerical issues with arccos
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        rotation_error = np.arccos(cos_angle)
        
        match = position_error < tolerance and rotation_error < tolerance
        
        results['test_cases'].append({
            'index': i,
            'position_error': float(position_error),
            'rotation_error': float(rotation_error),
            'match': match
        })
        
        results['max_position_error'] = max(results['max_position_error'], position_error)
        results['max_rotation_error'] = max(results['max_rotation_error'], rotation_error)
        
        if not match:
            results['all_match'] = False
    
    return results


def print_comparison_results(name, version, variant, results):
    """Print FK comparison results.
    
    :param name: Robot name
    :param version: Robot version
    :param variant: Robot variant
    :param results: Comparison results dictionary
    """
    model_id = f"{name}"
    if version:
        model_id += f" {version}"
    if variant:
        model_id += f" ({variant})"
    
    # Determine if we need to show structure trees
    show_structure = False
    if not results['success']:
        show_structure = True
    elif not results.get('dof_match', True):
        show_structure = True
    elif results.get('mjcf_end_link') != results.get('urdf_end_link'):
        show_structure = True
    elif not results.get('all_match', True):
        show_structure = True
        
    if not results['success']:
        beauty_print(f"⚠️  Error: {results.get('error', 'Unknown error')}", type="warning")
        if 'mjcf_end_link' in results:
            beauty_print(f"  MJCF end link: {results['mjcf_end_link']}", type="info")
        if 'urdf_end_link' in results:
            beauty_print(f"  URDF end link: {results['urdf_end_link']}", type="info")
        
        # Show structure trees
        if show_structure and 'mjcf_model' in results and 'urdf_model' in results:
            beauty_print(f"[MJCF] Structure:", type="info")
            results['mjcf_model'].print_tree(show_joints=True, show_fixed=False)
            print()
            beauty_print(f"[URDF] Structure:", type="info")
            results['urdf_model'].print_tree(show_joints=True, show_fixed=False)
        
        print()
        return
    
    # Show structure differences
    if not results.get('dof_match', True):
        beauty_print(f"⚠️  DOF mismatch: MJCF={results['num_dof_mjcf']}, URDF={results['num_dof_urdf']}", type="warning")
    
    if results.get('mjcf_end_link') != results.get('urdf_end_link'):
        beauty_print(f"⚠️  End link mismatch: MJCF={results['mjcf_end_link']}, URDF={results['urdf_end_link']}", type="warning")
        beauty_print(f"  Comparing at: MJCF={results['comparison_link_mjcf']}, URDF={results['comparison_link_urdf']}", type="info")
    
    if results['all_match']:
        beauty_print(f"✓ All {len(results['test_cases'])} test cases match (tolerance: 1e-5)", type="success")
        beauty_print(f"  Max position error: {results['max_position_error']:.2e} m", type="info")
        beauty_print(f"  Max rotation error: {results['max_rotation_error']:.2e} rad", type="info")
    else:
        beauty_print(f"⚠️  FK mismatch detected in {sum(1 for tc in results['test_cases'] if not tc.get('match', False))} out of {len(results['test_cases'])} test cases", type="warning")
        beauty_print(f"  Max position error: {results['max_position_error']:.2e} m", type="warning")
        beauty_print(f"  Max rotation error: {results['max_rotation_error']:.2e} rad", type="warning")
        
        # Show details of failed cases
        failed_cases = [tc for tc in results['test_cases'] if not tc.get('match', False)]
        if len(failed_cases) <= 5:
            for tc in failed_cases:
                if 'error' in tc:
                    beauty_print(f"  Test case {tc['index']}: {tc['error']}", type="warning")
                else:
                    beauty_print(f"  Test case {tc['index']}: pos_err={tc['position_error']:.2e} m, rot_err={tc['rotation_error']:.2e} rad", type="warning")
        else:
            beauty_print(f"  Showing first 5 failed cases:", type="warning")
            for tc in failed_cases[:5]:
                if 'error' in tc:
                    beauty_print(f"    Test case {tc['index']}: {tc['error']}", type="warning")
                else:
                    beauty_print(f"    Test case {tc['index']}: pos_err={tc['position_error']:.2e} m, rot_err={tc['rotation_error']:.2e} rad", type="warning")
    
    # Show structure trees if there are mismatches
    if show_structure and 'mjcf_model' in results and 'urdf_model' in results:
        print()
        beauty_print(f"[MJCF] Structure:", type="info")
        results['mjcf_model'].print_tree(show_joints=True, show_fixed=False)
        print()
        beauty_print(f"[URDF] Structure:", type="info")
        results['urdf_model'].print_tree(show_joints=True, show_fixed=False)
    
    print()


def main():
    """Main function to compare FK between URDF and MJCF models."""
    beauty_print("Collecting all available models from synriard...", type="info")
    all_models = get_all_models()
    
    # Group models by identity (name, version, variant)
    grouped = group_models_by_identity(all_models)
    
    # Filter to only models that have both URDF and MJCF
    common_models = {
        key: format_paths 
        for key, format_paths in grouped.items()
        if 'urdf' in format_paths and 'mjcf' in format_paths
    }
    
    beauty_print(f"Found {len(common_models)} models with both URDF and MJCF formats", type="success")
    print()
    
    if len(common_models) == 0:
        beauty_print("No common models found. Exiting.", type="warning")
        return
    
    # Compare FK for each common model
    for (name, version, variant), format_paths in sorted(common_models.items()):
        mjcf_path = format_paths['mjcf']
        urdf_path = format_paths['urdf']
        
        beauty_print(f"Model: {name} ({version} {variant})", type="module", centered=True)
        
        results = compare_fk(mjcf_path, urdf_path, num_test_cases=10)
        print_comparison_results(name, version, variant, results)


if __name__ == "__main__":
    main()
