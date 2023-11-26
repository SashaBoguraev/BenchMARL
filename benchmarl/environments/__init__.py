#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Task
from .pettingzoo.common import PettingZooTask
from .smacv2.common import Smacv2Task
from .vmas.common import VmasTask

# This is a registry mapping "envname/task_name" to the EnvNameTask.TASK_NAME enum
# It is used by automatically load task enums from yaml files
task_config_registry = {}
for env in [VmasTask, Smacv2Task, PettingZooTask]:
    env_config_registry = {
        f"{env.env_name()}/{task.name.lower()}": task for task in env
    }
    task_config_registry.update(env_config_registry)


from .pettingzoo.multiwalker import TaskConfig as MultiwalkerConfig
from .pettingzoo.simple_adverasary import TaskConfig as SimpleAdversaryConfig
from .pettingzoo.simple_crypto import TaskConfig as SimpleCryptoConfig
from .pettingzoo.simple_push import TaskConfig as SimplePushConfig
from .pettingzoo.simple_reference import TaskConfig as SimpleReferenceConfig
from .pettingzoo.simple_speaker_listener import (
    TaskConfig as SimpleSpeakerListenerConfig,
)
from .pettingzoo.simple_spread import TaskConfig as SimpleSpreadConfig
from .pettingzoo.simple_tag import TaskConfig as SimpleTagConfig
from .pettingzoo.simple_world_comm import TaskConfig as SimpleWorldComm
from .pettingzoo.waterworld import TaskConfig as WaterworldConfig
from .vmas.balance import TaskConfig as BalanceConfig
from .vmas.navigation import TaskConfig as NavigationConfig
from .vmas.sampling import TaskConfig as SamplingConfig
from .vmas.transport import TaskConfig as TransportConfig
from .vmas.wheel import TaskConfig as WheelConfig
from .vmas.simple_reference import TaskConfig as SimpleRefConfig
from .vmas.simple_reference_idiolect import TaskConfig as SimpleRefIdioConfig
from .vmas.simple_reference_idiolect_mem_buffer import TaskConfig as SimpleRefIdioMemConfig
from .vmas.simple_reference_idiolect_noise_mem import TaskConfig as SimpleRefIdioNoiseMemConfig
from .vmas.simple_reference_const import TaskConfig as ConstSimpleRefConfig
from .vmas.simple_reference_idiolect_const import TaskConfig as ConstSimpleRefIdioConfig
from .vmas.simple_reference_idiolect_mem_buffer_const import TaskConfig as ConstSimpleRefIdioMemConfig

# This is a registry mapping task config schemas names to their python dataclass
# It is used by hydra to validate loaded configs.
# You will see the "envname_taskname_config" strings in the hydra defaults at the top of yaml files.
# This feature is optional.
_task_class_registry = {
    "vmas_balance_config": BalanceConfig,
    "vmas_sampling_config": SamplingConfig,
    "vmas_navigation_config": NavigationConfig,
    "vmas_transport_config": TransportConfig,
    "vmas_wheel_config": WheelConfig,
    "vmas_simple_reference_config": SimpleRefConfig,
    "vmas_simple_reference_idiolect_config": SimpleRefIdioConfig,
    "vmas_simple_reference_idiolect_mem_buffer_config": SimpleRefIdioMemConfig,
    "vmas_simple_reference_idiolect_noise_mem_config": SimpleRefIdioNoiseMemConfig,
    "vmas_simple_reference_const_config": ConstSimpleRefConfig,
    "vmas_simple_reference_idiolect_const_config": ConstSimpleRefIdioConfig,
    "vmas_simple_reference_idiolect_mem_buffer_const_config": ConstSimpleRefIdioMemConfig,
    "pettingzoo_multiwalker_config": MultiwalkerConfig,
    "pettingzoo_waterworld_config": WaterworldConfig,
    "pettingzoo_simple_adversary_config": SimpleAdversaryConfig,
    "pettingzoo_simple_crypto_config": SimpleCryptoConfig,
    "pettingzoo_simple_push_config": SimplePushConfig,
    "pettingzoo_simple_reference_config": SimpleReferenceConfig,
    "pettingzoo_simple_speaker_listener_config": SimpleSpeakerListenerConfig,
    "pettingzoo_simple_spread_config": SimpleSpreadConfig,
    "pettingzoo_simple_tag_config": SimpleTagConfig,
    "pettingzoo_simple_world_comm_config": SimpleWorldComm,
}
