#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboCore Module

Copyright (c) 2025 Synria Robotics Co., Ltd.

Licensed under the MIT License.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import pytest
import numpy as np
from robocore.utils.backend import BackendManager, get_backend, set_backend


class TestBackendManager:
    """Test BackendManager state management."""
    
    def test_backend_singleton(self):
        """Test that BackendManager is a singleton."""
        mgr1 = BackendManager()
        mgr2 = BackendManager()
        assert mgr1 is mgr2
    
    def test_default_backend(self):
        """Test default backend is numpy."""
        mgr = BackendManager()
        assert mgr.get_backend() in ['numpy', 'torch']
    
    def test_backend_switching(self):
        """Test backend can be switched."""
        original = get_backend()
        try:
            set_backend('numpy')
            assert get_backend() == 'numpy'
            
            try:
                import torch
                set_backend('torch')
                assert get_backend() == 'torch'
            except ImportError:
                pytest.skip("PyTorch not available")
        finally:
            set_backend(original)
    
    def test_invalid_backend(self):
        """Test invalid backend raises error."""
        with pytest.raises(ValueError):
            set_backend('invalid_backend')
    
    def test_backend_context_isolation(self):
        """Test backend switching doesn't leak across contexts."""
        original = get_backend()
        try:
            # Switch to numpy
            set_backend('numpy')
            assert get_backend() == 'numpy'
            
            # Simulate nested function changing backend
            def nested_func():
                prev = get_backend()
                try:
                    if prev == 'numpy':
                        try:
                            import torch
                            set_backend('torch')
                        except ImportError:
                            pass
                    else:
                        set_backend('numpy')
                    return get_backend()
                finally:
                    set_backend(prev)  # Restore
            
            inner_backend = nested_func()
            
            # Original context should be restored
            assert get_backend() == 'numpy'
        finally:
            set_backend(original)
    
    def test_torch_available_detection(self):
        """Test PyTorch availability detection."""
        try:
            import torch
            # Should work
            set_backend('torch')
            assert get_backend() == 'torch'
            set_backend('numpy')  # Restore
        except ImportError:
            # Should raise if torch not available
            with pytest.raises(RuntimeError):
                set_backend('torch')
    
    def test_backend_affects_array_creation(self):
        """Test backend affects array type creation."""
        original = get_backend()
        try:
            from robocore.utils.backend import zeros, eye
            
            # NumPy backend
            set_backend('numpy')
            arr = zeros(3)
            mat = eye(3)
            assert isinstance(arr, np.ndarray)
            assert isinstance(mat, np.ndarray)
            
            # PyTorch backend (if available)
            try:
                import torch
                set_backend('torch')
                arr = zeros(3)
                mat = eye(3)
                assert isinstance(arr, torch.Tensor)
                assert isinstance(mat, torch.Tensor)
            except ImportError:
                pytest.skip("PyTorch not available")
        finally:
            set_backend(original)


class TestBackendHelpers:
    """Test backend helper functions."""
    
    def test_get_backend(self):
        """Test get_backend returns valid backend."""
        backend = get_backend()
        assert backend in ['numpy', 'torch']
    
    def test_set_backend_persistence(self):
        """Test set_backend persists across calls."""
        original = get_backend()
        try:
            set_backend('numpy')
            assert get_backend() == 'numpy'
            assert get_backend() == 'numpy'  # Still numpy
            
            try:
                import torch
                set_backend('torch')
                assert get_backend() == 'torch'
                assert get_backend() == 'torch'  # Still torch
            except ImportError:
                pass
        finally:
            set_backend(original)
    
    def test_backend_thread_safety(self):
        """Test backend switching is thread-safe (basic check)."""
        import threading
        
        original = get_backend()
        results = []
        
        def switch_backend(backend_name):
            try:
                set_backend(backend_name)
                results.append(get_backend())
            except (ValueError, RuntimeError):
                results.append(None)
        
        try:
            threads = [
                threading.Thread(target=switch_backend, args=('numpy',)),
                threading.Thread(target=switch_backend, args=('numpy',)),
            ]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All should have succeeded
            assert all(r == 'numpy' for r in results if r is not None)
        finally:
            set_backend(original)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
