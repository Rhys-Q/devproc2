"""Module containers for the nn frontend."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Optional

from devproc2.nn.specs import Parameter, with_parameter_name


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_is_tracing", False)

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Parameter) and self._is_tracing:
            raise RuntimeError("cannot add Parameter while tracing/building a Module")
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        if isinstance(value, Module):
            self._modules[name] = value
            self._parameters.pop(name, None)
        elif isinstance(value, Parameter):
            self._parameters[name] = value
            self._modules.pop(name, None)
        else:
            self._modules.pop(name, None)
            self._parameters.pop(name, None)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")

    def named_modules(
        self,
        prefix: str = "",
        memo: Optional[set[int]] = None,
    ) -> Iterator[tuple[str, "Module"]]:
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(child_prefix, memo)

    def named_parameters(
        self,
        prefix: str = "",
        memo: Optional[set[int]] = None,
    ) -> Iterator[tuple[str, Parameter]]:
        if memo is None:
            memo = set()
        for name, param in self._parameters.items():
            if id(param) in memo:
                continue
            memo.add(id(param))
            path = f"{prefix}.{name}" if prefix else name
            yield path, with_parameter_name(param, path)
        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(child_prefix, memo)

    def state_dict(self) -> dict[str, Parameter]:
        return dict(self.named_parameters())

    def _set_tracing_recursive(self, value: bool, memo: Optional[set[int]] = None) -> None:
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        object.__setattr__(self, "_is_tracing", value)
        for module in self._modules.values():
            module._set_tracing_recursive(value, memo)

    def _assign_parameter_names(
        self,
        prefix: str = "",
        module_memo: Optional[set[int]] = None,
        param_memo: Optional[set[int]] = None,
    ) -> None:
        if module_memo is None:
            module_memo = set()
        if param_memo is None:
            param_memo = set()
        if id(self) in module_memo:
            return
        module_memo.add(id(self))
        for name, param in tuple(self._parameters.items()):
            if id(param) in param_memo:
                continue
            param_memo.add(id(param))
            path = f"{prefix}.{name}" if prefix else name
            named = with_parameter_name(param, path)
            object.__setattr__(self, name, named)
            self._parameters[name] = named
        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            module._assign_parameter_names(child_prefix, module_memo, param_memo)


class ModuleList(Module):
    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        self._items: list[Module] = []
        if modules is not None:
            for module in modules:
                self.append(module)

    def append(self, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError("ModuleList.append expects a Module")
        idx = len(self._items)
        self._items.append(module)
        setattr(self, str(idx), module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Module:
        return self._items[idx]


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.layers = ModuleList(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


__all__ = [
    "Module",
    "ModuleList",
    "Sequential",
]
