try:
    import slime_plugins.megatron_bridge.glm4v_moe  # noqa: F401  # register GLM-4.6V bridge
except ImportError as _e:
    import warnings

    warnings.warn(
        f"slime_plugins.megatron_bridge.glm4v_moe failed to import "
        f"(GLM-4.6V bridge will be unavailable): {_e}"
    )
