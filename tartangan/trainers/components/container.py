from .base import TrainerComponent


class ComponentContainer(TrainerComponent):
    """A component that contains other components."""
    def __init__(self, *components):
        self.components = components

    def add_components(self, *components):
        for component in components:
            component.trainer = self.trainer
        self.components += components

    def invoke(self, hook_name, *args, **kwargs):
        hook_f_name = f'on_{hook_name}'
        for component in self.components:
            hook = getattr(component, hook_f_name, None)
            hook(*args, **kwargs)
