import os
import tkinter as tk
import unittest
import uuid

from single_instance import SingleInstanceGuard


@unittest.skipUnless(os.name == "nt", "SingleInstanceGuard usa a API Win32")
class SingleInstanceGuardTests(unittest.TestCase):
    def test_second_guard_detects_existing_mutex_and_exit_releases_it(self):
        mutex_name = rf"Local\MemoryMeet.Test.{uuid.uuid4()}"
        first = SingleInstanceGuard(mutex_name=mutex_name, window_title=None)
        second = SingleInstanceGuard(mutex_name=mutex_name, window_title=None)

        try:
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
        finally:
            second.close()
            first.close()

        after_close = SingleInstanceGuard(mutex_name=mutex_name, window_title=None)
        try:
            self.assertTrue(after_close.acquire())
        finally:
            after_close.close()

    def test_acquire_is_idempotent_for_the_owner(self):
        guard = SingleInstanceGuard(
            mutex_name=rf"Local\MemoryMeet.Test.{uuid.uuid4()}",
            window_title=None,
        )
        try:
            self.assertTrue(guard.acquire())
            self.assertTrue(guard.acquire())
        finally:
            guard.close()

    def test_second_guard_restores_existing_window(self):
        unique = str(uuid.uuid4())
        mutex_name = rf"Local\MemoryMeet.Test.{unique}"
        window_title = f"MemoryMeet Test {unique}"
        first = SingleInstanceGuard(mutex_name=mutex_name, window_title=window_title)
        second = SingleInstanceGuard(
            mutex_name=mutex_name,
            window_title=window_title,
            activation_timeout=0.5,
        )
        root = tk.Tk()
        root.title(window_title)
        root.update()

        try:
            self.assertTrue(first.acquire())
            root.iconify()
            root.update()
            self.assertEqual(root.state(), "iconic")

            self.assertFalse(second.acquire())
            root.update()
            self.assertNotEqual(root.state(), "iconic")
        finally:
            second.close()
            first.close()
            root.destroy()


if __name__ == "__main__":
    unittest.main()
