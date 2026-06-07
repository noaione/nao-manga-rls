"""
MIT License

Copyright (c) 2022-present noaione

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Custom solution to load WinML, compatible with Python 3.14

from __future__ import annotations

import ctypes
import platform
import sys
from pathlib import Path
from typing import Optional

__all__ = (
    "find_metapackage",
    "get_winml_ep_libraries",
)

# Enums
WinMLEpReadyState_Ready = 0
WinMLEpReadyState_NotReady = 1
WinMLEpReadyState_NotPresent = 2

WinMLEpCertification_Unknown = 0
WinMLEpCertification_Certified = 1
WinMLEpCertification_Uncertified = 2

# Keep DLL handles alive so the loader doesn't unmap them from under ORT
_preloaded_dlls: list[ctypes.CDLL] = []


class WinMLEpInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("version", ctypes.c_char_p),
        ("packageFamilyName", ctypes.c_char_p),
        ("libraryPath", ctypes.c_char_p),
        ("packageRootPath", ctypes.c_char_p),
        ("readyState", ctypes.c_int),
        ("certification", ctypes.c_int),
    ]


def win():
    return sys.platform == "win32"


def find_metapackage() -> Optional[Path]:
    if not win():
        return None

    from winrt.windows.applicationmodel import Package
    from winrt.windows.management.deployment import PackageManager
    from winrt.windows.system import ProcessorArchitecture

    matching_arch = None
    match platform.machine().upper():
        case "AMD64":
            matching_arch = ProcessorArchitecture.X64
        case "i386":
            matching_arch = ProcessorArchitecture.X86
        case "ARM64":
            matching_arch = ProcessorArchitecture.ARM64
        case "ARM":
            matching_arch = ProcessorArchitecture.ARM

    if not matching_arch:
        return None

    pm = PackageManager()
    packages = pm.find_packages_by_user_security_id("")
    candidates: list[Package] = []
    for pkg in packages:
        if "WindowsAppRuntime.2" in pkg.display_name and pkg.id.architecture == matching_arch:
            candidates.append(pkg)

    candidates.sort(
        key=lambda p: (
            p.id.version.major,
            p.id.version.minor,
            p.id.version.build,
            p.id.version.revision,
        ),
        reverse=True,
    )

    for pkg in candidates:
        full_path = Path(pkg.installed_path)
        ai_ml_dll = full_path / "Microsoft.Windows.AI.MachineLearning.dll"
        if ai_ml_dll.exists():
            return full_path


def get_winml_ep_libraries() -> list[tuple[str, Path]]:
    if not win():
        return []

    import ctypes.wintypes

    wasdk_path = find_metapackage()
    if not wasdk_path:
        return []

    try:
        winml = ctypes.WinDLL(wasdk_path / "Microsoft.Windows.AI.MachineLearning.dll")
    except OSError:
        return []

    WinMLEpCatalogCreate = winml.WinMLEpCatalogCreate
    WinMLEpCatalogCreate.restype = ctypes.HRESULT
    WinMLEpCatalogCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    WinMLEpCatalogRelease = winml.WinMLEpCatalogRelease
    WinMLEpCatalogRelease.restype = None
    WinMLEpCatalogRelease.argtypes = [ctypes.c_void_p]

    WinMLEpCatalogEnumProviders = winml.WinMLEpCatalogEnumProviders
    WinMLEpCatalogEnumProviders.restype = ctypes.HRESULT
    WinMLEpCatalogEnumProviders.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    WinMLEpEnsureReady = winml.WinMLEpEnsureReady
    WinMLEpEnsureReady.restype = ctypes.HRESULT
    WinMLEpEnsureReady.argtypes = [ctypes.c_void_p]
    WinMLEpGetReadyState = winml.WinMLEpGetReadyState
    WinMLEpGetReadyState.restype = ctypes.HRESULT
    WinMLEpGetReadyState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    WinMLEpGetLibraryPathSize = winml.WinMLEpGetLibraryPathSize
    WinMLEpGetLibraryPathSize.restype = ctypes.HRESULT
    WinMLEpGetLibraryPathSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]

    WinMLEpGetLibraryPath = winml.WinMLEpGetLibraryPath
    WinMLEpGetLibraryPath.restype = ctypes.HRESULT
    WinMLEpGetLibraryPath.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]

    ENUM_CALLBACK = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL,
        ctypes.c_void_p,  # WinMLEpHandle ep
        ctypes.POINTER(WinMLEpInfo),  # const WinMLEpInfo* info
        ctypes.c_void_p,  # void* context
    )

    catalog = ctypes.c_void_p()
    if WinMLEpCatalogCreate(ctypes.byref(catalog)) != 0:
        return []

    registered: list[tuple[str, Path]] = []

    def _ep_callback(ep_handle, info_ptr, _ctx):
        info = info_ptr.contents

        if info.certification != WinMLEpCertification_Certified:
            return True
        if info.readyState == WinMLEpReadyState_NotPresent:
            return True
        if WinMLEpEnsureReady(ep_handle) != 0:
            return True

        ready_state = ctypes.c_int()
        WinMLEpGetReadyState(ep_handle, ctypes.byref(ready_state))

        path_size = ctypes.c_size_t()
        if WinMLEpGetLibraryPathSize(ep_handle, ctypes.byref(path_size)) != 0 or path_size.value == 0:
            return True

        path_buf = ctypes.create_string_buffer(path_size.value)
        used = ctypes.c_size_t()
        if WinMLEpGetLibraryPath(ep_handle, path_size.value, path_buf, ctypes.byref(used)) != 0:
            return True

        name = info.name.decode() if info.name else None
        path = path_buf.value.decode()

        if name and path:
            full_path = Path(path)
            registered.append((name, full_path))
        return True

    try:
        WinMLEpCatalogEnumProviders(catalog, ENUM_CALLBACK(_ep_callback), None)
    finally:
        WinMLEpCatalogRelease(catalog)
    return registered
