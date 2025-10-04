"""3D transformation library with unified numpy/torch backend."""

from .se3 import (
    make_transform, translation_transform, rotation_transform,
    get_rotation, get_translation,
    transform_multiply, transform_inverse, transform_apply, transform_interpolate,
)

from .so3 import (
    rpy_to_matrix, axis_angle_to_matrix, quaternion_to_matrix, euler_to_matrix,
    rotation_x, rotation_y, rotation_z,
    rotation_multiply, rotation_inverse, rotation_apply,
    skew_symmetric, rotation_from_vectors,
)

from .conversions import (
    matrix_to_rpy, matrix_to_axis_angle, matrix_to_quaternion, matrix_to_euler,
    quaternion_to_rpy, quaternion_to_axis_angle,
    quaternion_normalize, quaternion_conjugate, quaternion_inverse, quaternion_multiply,
    rpy_to_quaternion, rpy_to_axis_angle,
    axis_angle_to_quaternion, axis_angle_to_rpy,
    axis_angle_to_compact, compact_to_axis_angle,
)

from .utils import (
    slerp, rotation_interpolate,
    rotation_distance, rotation_error, orientation_error,
    is_rotation_matrix, is_transform_matrix, look_at,
)
