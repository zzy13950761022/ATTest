from typing import List, Dict, Any, Optional, Iterator
import os
import json


class ChatResponse:
    """Response from LLM chat."""
    def __init__(
        self,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: str = "",
        role: Optional[str] = None,  # role is ignored but accepted for backward compatibility
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content or ""
        
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """
    OpenAI-compatible LLM client with function calling support.
    Supports any API that follows the OpenAI format (DeepSeek, OpenAI, vLLM, etc.)
    """

    def __init__(
        self, 
        base_url: str = "", 
        api_key: str = "", 
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 180,
        debug_log: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        env_debug = os.environ.get("TESTAGENT_DEBUG_LLM", "").lower()
        self.debug_log = debug_log or env_debug in ("1", "true", "yes", "y", "on")
        
        # Try to import openai library
        try:
            import openai
            self.openai = openai
            self._use_openai_lib = True
        except ImportError:
            self._use_openai_lib = False
            import requests
            self.requests = requests

    def _log_debug(self, title: str, data: Any) -> None:
        """Print debug information when enabled."""
        if not self.debug_log:
            return
        try:
            pretty = json.dumps(data, ensure_ascii=False, indent=2)
        except TypeError:
            pretty = str(data)
        print(f"[LLM DEBUG] {title}:\n{pretty}\n")

    def _truncate_messages_for_log(
        self, messages: List[Dict[str, Any]], max_length: int = 2000
    ) -> List[Dict[str, Any]]:
        truncated: List[Dict[str, Any]] = []
        for msg in messages:
            item = dict(msg)
            content = item.get("content")
            if isinstance(content, str) and len(content) > max_length:
                item["content"] = (
                    f"{content[:max_length]}... (truncated {len(content) - max_length} chars)"
                )
            truncated.append(item)
        return truncated

    def chat(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None
    ) -> ChatResponse:
        """
        Send chat request to LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions in OpenAI format
            
        Returns:
            ChatResponse with content and optional tool_calls
        """
        if self._use_openai_lib:
            return self._chat_with_openai_lib(messages, tools)
        else:
            return self._chat_with_requests(messages, tools)

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize messages for target model.
        - deepseek-reasoner: 保留 reasoning_content（助手消息缺失则补空字符串）
        - 其他模型: 移除 reasoning_content，避免不支持的字段导致 400
        """
        requires_reasoning = "deepseek-reasoner" in (self.model or "").lower()
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            m = dict(msg)  # shallow copy
            if requires_reasoning:
                if m.get("role") == "assistant":
                    m.setdefault("reasoning_content", "")
            else:
                m.pop("reasoning_content", None)
            normalized.append(m)
        return normalized
    
    def _chat_with_openai_lib(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None
    ) -> ChatResponse:
        """Use OpenAI library for API calls."""
        client = self.openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        kwargs = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        
        if tools:
            kwargs["tools"] = tools

        self._log_debug(
            "Request",
            {
                "base_url": self.base_url,
                "model": self.model,
                "messages": self._truncate_messages_for_log(kwargs["messages"]),
                "tools": tools or [],
            },
        )
            
        try:
            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            
            content = message.content or ""
            tool_calls = []
            reasoning_content = getattr(message, "reasoning_content", "") or ""
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            self._log_debug(
                "Response",
                {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                },
            )

            return ChatResponse(content, tool_calls, reasoning_content)
            
        except Exception as e:
            raise RuntimeError(f"LLM API error: {e}")
    
    def _chat_with_requests(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None
    ) -> ChatResponse:
        """Use requests library for API calls (fallback)."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if tools:
            payload["tools"] = tools

        self._log_debug(
            "Request",
            {
                "url": url,
                "model": self.model,
                "messages": self._truncate_messages_for_log(payload["messages"]),
                "tools": tools or [],
            },
        )
        
        try:
            response = self.requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            message = data["choices"][0]["message"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            reasoning_content = message.get("reasoning_content", "") or ""

            self._log_debug(
                "Response",
                {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                },
            )
            
            return ChatResponse(content, tool_calls, reasoning_content)
            
        except Exception as e:
            raise RuntimeError(f"LLM API error: {e}")

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        Stream chat response (for future use).
        Currently not implemented - returns full response.
        """
        response = self.chat(messages)
        yield response.content
