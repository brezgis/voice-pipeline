#!/usr/bin/env python3
"""
Clawdbot LLM Service for Pipecat

This service integrates with Clawdbot's agent system to provide LLM functionality
while preserving the full Claude session context (SOUL.md, memory, tools, identity).

Instead of using a direct Anthropic API integration, this routes through:
  clawdbot agent --session-id voice --json --timeout 30

This ensures the voice pipeline uses the same Claude instance as text chats,
with full access to workspace context, tools, and memory.
"""

import asyncio
import subprocess
import json
from typing import Optional, Dict, Any, AsyncIterator
from pydantic import BaseModel

# Pipecat imports
from pipecat.frames.frames import (
    Frame, 
    LLMMessagesFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    SystemFrame
)
from pipecat.services.llm_service import LLMService


class ClawdbotLLMService(LLMService):
    """
    LLM Service that routes requests through Clawdbot agent subprocess
    
    This maintains the full Clawdbot context and session state while
    providing streaming responses compatible with Pipecat's pipeline.
    """
    
    def __init__(
        self, 
        session_id: str = "voice",
        timeout: int = 30,
        model: str = "claude-sonnet-4",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.timeout = timeout  
        self.model = model
        self._current_process: Optional[subprocess.Popen] = None
        
    async def _send_prompt(self, prompt: str) -> AsyncIterator[str]:
        """
        Send prompt to Clawdbot agent and yield streaming response chunks
        """
        cmd = [
            'clawdbot', 'agent', 
            '--session-id', self.session_id,
            '--json',
            '--timeout', str(self.timeout),
            '--model', self.model
        ]
        
        try:
            # Start clawdbot agent subprocess
            self._current_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered for real-time streaming
            )
            
            # Send the prompt
            self._current_process.stdin.write(prompt + '\n')
            self._current_process.stdin.flush()
            self._current_process.stdin.close()
            
            # Read streaming response
            full_response = ""
            while True:
                line = self._current_process.stdout.readline()
                if not line:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Parse JSON response from clawdbot
                    response_data = json.loads(line)
                    
                    if response_data.get('type') == 'content':
                        # Content chunk
                        chunk = response_data.get('text', '')
                        if chunk:
                            full_response += chunk
                            yield chunk
                            
                    elif response_data.get('type') == 'error':
                        # Error from clawdbot
                        error_msg = response_data.get('message', 'Unknown error')
                        print(f"Clawdbot error: {error_msg}")
                        break
                        
                    elif response_data.get('type') == 'done':
                        # Response complete
                        break
                        
                except json.JSONDecodeError:
                    # Non-JSON line, might be status or error
                    print(f"Clawdbot non-JSON output: {line}")
                    continue
            
            # Wait for process to complete
            await asyncio.create_task(asyncio.to_thread(self._current_process.wait))
            
            if not full_response.strip():
                yield "I apologize, but I encountered an issue processing your request."
                
        except Exception as e:
            print(f"Error communicating with Clawdbot: {e}")
            yield f"I apologize, but I encountered a technical issue: {str(e)}"
            
        finally:
            self._current_process = None
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames from the pipeline"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            # Convert messages to prompt and get response
            prompt = self._messages_to_prompt(frame.messages)
            
            # Stream response chunks
            async for chunk in self._send_prompt(prompt):
                if chunk.strip():
                    # Send text chunk frame
                    await self.push_frame(
                        LLMTextFrame(text=chunk),
                        direction
                    )
                    
            # Send end frame
            await self.push_frame(LLMFullResponseEndFrame(), direction)
            
        elif isinstance(frame, TextFrame):
            # Direct text input
            async for chunk in self._send_prompt(frame.text):
                if chunk.strip():
                    await self.push_frame(
                        LLMTextFrame(text=chunk),
                        direction
                    )
                    
            await self.push_frame(LLMFullResponseEndFrame(), direction)
    
    def _messages_to_prompt(self, messages: list) -> str:
        """
        Convert LLM messages format to a simple prompt
        
        Clawdbot agent handles the conversation context internally,
        so we just need to send the latest user message.
        """
        if not messages:
            return "Hello"
            
        # Get the last user message
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                return msg.get('content', '')
        
        # Fallback: join all content
        content_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if content:
                content_parts.append(f"{role}: {content}")
                
        return '\n'.join(content_parts) if content_parts else "Hello"
    
    async def stop(self):
        """Stop the service and cleanup"""
        if self._current_process:
            try:
                self._current_process.terminate()
                await asyncio.create_subprocess_wait(self._current_process)
            except:
                pass  # Process might already be dead
            self._current_process = None
    
    def can_generate_metrics(self) -> bool:
        """Whether this service can generate metrics"""
        return True


# Test function  
async def test_clawdbot_service():
    """Test the Clawdbot LLM service"""
    print("=== Testing Clawdbot LLM Service ===")
    
    service = ClawdbotLLMService(session_id="voice-test")
    
    # Test with a simple prompt
    print("Testing prompt: 'Hello, how are you?'")
    
    response_chunks = []
    async for chunk in service._send_prompt("Hello, how are you?"):
        print(f"Chunk: {repr(chunk)}")
        response_chunks.append(chunk)
    
    full_response = ''.join(response_chunks)
    print(f"Full response: {full_response}")
    
    await service.stop()
    print("Test complete")

if __name__ == "__main__":
    asyncio.run(test_clawdbot_service())