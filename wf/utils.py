import functools
from typing import List

from flytekit import task
from flytekitplugins.pod import Pod
from kubernetes.client.models import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1SecurityContext,
    V1Volume,
    V1VolumeMount,
)
from latch.resources.tasks import (
    _get_l40s_pod,
    _get_large_gpu_pod,
    _get_small_gpu_pod,
    get_v100_x1_pod,
)


def _add_privileged_and_shm(x: Pod):
    # Add privileged access
    containers = x.pod_spec.containers
    assert containers is not None
    assert len(containers) > 0
    container: V1Container = containers[0]
    container.security_context = V1SecurityContext(privileged=True)

    # Add shared memory
    name = "shared-memory"
    spec = x.pod_spec

    if spec.volumes is None:
        spec.volumes = []
    vols: List[V1Volume] = spec.volumes
    vols.append(
        V1Volume(
            name=name,
            empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="11Gi"),
        )
    )

    if container.volume_mounts is None:
        container.volume_mounts = []
    mounts: List[V1VolumeMount] = container.volume_mounts
    mounts.append(V1VolumeMount(mount_path="/dev/shm", name=name))

    return x


privileged_largedisk_v100_x1_gpu_task = functools.partial(
    task, task_config=_add_privileged_and_shm(get_v100_x1_pod())
)

privileged_largedisk_large_gpu_task = functools.partial(
    task, task_config=_add_privileged_and_shm(_get_large_gpu_pod())
)

privileged_largedisk_small_gpu_task = functools.partial(
    task, task_config=_add_privileged_and_shm(_get_small_gpu_pod())
)

privileged_largedisk_g6e_8xlarge_task = functools.partial(
    task,
    task_config=_add_privileged_and_shm(
        _get_l40s_pod("g6e-8xlarge", cpu=32, memory_gib=256, gpus=1)
    ),
)
