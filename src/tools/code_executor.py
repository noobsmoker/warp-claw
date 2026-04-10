"""
Code Executor Tool
Sandboxed Python/Shell execution for Warp-Claw.
"""

import asyncio
import tempfile
import os
import sys
import json
from typing import Dict, Any
from datetime import datetime

from .base_tool import BaseTool, ToolResult, ToolRegistry


class CodeExecutor(BaseTool):
    """
    Execute Python or Shell code in a sandboxed environment.
    """
    
    name = "execute_python"
    description = "Execute Python code in a sandboxed environment. Use for calculations, data processing, and code execution."
    category = "code"
    tags = ["python", "execution", "sandbox", "code"]
    default_timeout = 30
    
    # Restricted imports for security
    BLOCKED_MODULES = {
        "os": ["fork", "spawn", "system", "popen"],
        "subprocess": ["spawn", "Popen"],
        "socket": ["socket", "create_connection"],
        "requests": ["*"],
        "urllib": ["*"],
        "http": ["*"],
        "ftplib": ["*"],
        "telnetlib": ["*"],
        "pty": ["*"],
        "tty": ["*"],
        "termios": ["*"],
        "crypt": ["*"],
        "pwd": ["*"],
        "grp": ["*"],
    }
    
    def __init__(self):
        super().__init__()
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up the sandboxed environment."""
        # Create temp directory
        self._temp_dir = tempfile.mkdtemp(prefix="warp_claw_")
    
    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        **kwargs
    ) -> ToolResult:
        """
        Execute code.
        
        Args:
            code: Code to execute
            language: "python" or "shell"
            timeout: Execution timeout in seconds
            
        Returns:
            ToolResult with stdout, stderr, returncode
        """
        if language not in ("python", "shell"):
            return ToolResult(
                success=False,
                result=None,
                error=f"Unsupported language: {language}"
            )
        
        if language == "python":
            return await self._execute_python(code, timeout)
        else:
            return await self._execute_shell(code, timeout)
    
    async def _execute_python(self, code: str, timeout: int) -> ToolResult:
        """Execute Python code."""
        # Security check
        security_error = self._check_security(code)
        if security_error:
            return ToolResult(
                success=False,
                result=None,
                error=security_error
            )
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            dir=self._temp_dir
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Run Python
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",  # Unbuffered
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_sandbox_env()
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            
            return ToolResult(
                success=proc.returncode == 0,
                result={
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "returncode": proc.returncode
                },
                execution_time_ms=0,
                tokens_estimate=len(code) // 4
            )
            
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(
                success=False,
                result=None,
                error=f"Execution timeout after {timeout}s"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    async def _execute_shell(self, code: str, timeout: int) -> ToolResult:
        """Execute shell code."""
        # Very restricted shell execution
        allowed_commands = {"echo", "date", "whoami", "pwd", "ls", "cat"}
        
        parts = code.strip().split()
        if not parts or parts[0] not in allowed_commands:
            return ToolResult(
                success=False,
                result=None,
                error=f"Shell command not allowed. Allowed: {allowed_commands}"
            )
        
        try:
            proc = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            
            return ToolResult(
                success=proc.returncode == 0,
                result={
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "returncode": proc.returncode
                }
            )
            
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(
                success=False,
                result=None,
                error=f"Execution timeout after {timeout}s"
            )
    
    def _check_security(self, code: str) -> Optional[str]:
        """Check code for dangerous patterns."""
        dangerous = [
            ("import os", "import os"),
            ("import subprocess", "import subprocess"),
            ("import socket", "import socket"),
            ("__import__", "__import__"),
            ("eval(", "eval()"),
            ("exec(", "exec()"),
            ("open(", "file open"),
            ("write(", "file write"),
            ("requests", "external HTTP"),
            ("urllib", "external HTTP"),
        ]
        
        for pattern, name in dangerous:
            if pattern in code:
                return f"Blocked: {name} is not allowed"
        
        return None
    
    def _get_sandbox_env(self) -> Dict[str, str]:
        """Get sandboxed environment variables."""
        env = os.environ.copy()
        env["PYTHONPATH"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # Remove potentially dangerous vars
        for key in list(env.keys()):
            if key.startswith("PYTHON") and key != "PYTHONUNBUFFERED":
                del env[key]
        return env
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "shell"],
                        "description": "Language to execute",
                        "default": "python"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        }


# Register the tool
registry = CodeExecutor()
ToolRegistry.register(registry)