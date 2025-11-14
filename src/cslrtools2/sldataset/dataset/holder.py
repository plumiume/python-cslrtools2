# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type guards for sign language dataset keys.

This module provides the SLKeyHolder class for runtime type checking
of different key types used in sign language datasets.
"""

from __future__ import annotations

from typing import TypeGuard


class SLKeyHolder[Kmeta: str, Kvid: str, Klm: str, Ktgt: str]:
    """Type guards for sign language dataset keys.
    
    Provides runtime type checking for the different key types used in
    sign language datasets (metadata, video, landmark, target).
    
    Type Parameters:
        Kmeta: String type for metadata keys.
        Kvid: String type for video data keys.
        Klm: String type for landmark data keys.
        Ktgt: String type for target/label keys.
    """

    @classmethod
    def is_metadata_key(cls, obj: object) -> TypeGuard[Kmeta]:
        """Check if object is a valid metadata key.
        
        Args:
            obj (:obj:`object`): Object to check.
            
        Returns:
            :obj:`bool`: :obj:`True` if obj is a valid metadata key.
        """
        return isinstance(obj, str)

    @classmethod
    def is_video_key(cls, obj: object) -> TypeGuard[Kvid]:
        """Check if object is a valid video key.
        
        Args:
            obj (:obj:`object`): Object to check.
            
        Returns:
            :obj:`bool`: :obj:`True` if obj is a valid video key.
        """
        return isinstance(obj, str)
    
    @classmethod
    def is_landmark_key(cls, obj: object) -> TypeGuard[Klm]:
        """Check if object is a valid landmark key.
        
        Args:
            obj (:obj:`object`): Object to check.
            
        Returns:
            :obj:`bool`: :obj:`True` if obj is a valid landmark key.
        """
        return isinstance(obj, str)
    
    @classmethod
    def is_target_key(cls, obj: object) -> TypeGuard[Ktgt]:
        """Check if object is a valid target key.
        
        Args:
            obj (:obj:`object`): Object to check.
            
        Returns:
            :obj:`bool`: :obj:`True` if obj is a valid target key.
        """
        return isinstance(obj, str)
