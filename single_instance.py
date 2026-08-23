"""Garantia de instância única e ativação da janela existente no Windows."""

import ctypes
from ctypes import wintypes
import time
from typing import Optional


APP_TITLE = "MemoryMeet"
MUTEX_NAME = r"Local\MemoryMeet.SingleInstance"

ERROR_ALREADY_EXISTS = 183
SW_SHOW = 5
SW_RESTORE = 9


class SingleInstanceGuard:
    """Mantém um mutex nomeado vivo enquanto esta for a instância principal."""

    def __init__(
        self,
        mutex_name: str = MUTEX_NAME,
        window_title: Optional[str] = APP_TITLE,
        activation_timeout: float = 5.0,
    ):
        self.mutex_name = mutex_name
        self.window_title = window_title
        self.activation_timeout = activation_timeout
        self._handle = None

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
        self._kernel32.CreateMutexW.restype = wintypes.HANDLE
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def acquire(self) -> bool:
        """Retorna True para a primeira instância; ativa a existente e retorna False nas demais."""
        if self._handle is not None:
            return True

        # CreateMutex só garante ERROR_ALREADY_EXISTS quando o objeto já existia;
        # zerar antes evita interpretar um last-error antigo como resultado desta chamada.
        ctypes.set_last_error(0)
        handle = self._kernel32.CreateMutexW(None, False, self.mutex_name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())

        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            self._kernel32.CloseHandle(handle)
            self._activate_existing_window()
            return False

        self._handle = handle
        return True

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None

    def _activate_existing_window(self) -> bool:
        if not self.window_title:
            return False

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
        user32.FindWindowW.restype = wintypes.HWND
        user32.IsIconic.argtypes = (wintypes.HWND,)
        user32.IsIconic.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.ShowWindow.restype = wintypes.BOOL
        user32.BringWindowToTop.argtypes = (wintypes.HWND,)
        user32.BringWindowToTop.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        user32.SetForegroundWindow.restype = wintypes.BOOL

        deadline = time.monotonic() + self.activation_timeout
        while True:
            hwnd = user32.FindWindowW(None, self.window_title)
            if hwnd:
                command = SW_RESTORE if user32.IsIconic(hwnd) else SW_SHOW
                user32.ShowWindow(hwnd, command)
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
