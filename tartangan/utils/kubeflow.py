def key_to_kf_name(k):
    """Convert a name to something Kubeflow likes."""
    k = k.replace('_', '-').lower()
    return k
