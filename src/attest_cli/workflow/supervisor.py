"""
Supervisor Agent - Interprets user feedback and decides workflow actions.
"""
import json
from dataclasses import dataclass
from typing import Optional, Dict, List

from ..llm import LLMClient
from .state import WorkflowState


@dataclass
class Action:
    """
    Represents a workflow action decided by the supervisor.
    """
    type: str  # "continue", "retry", "goto", "quit"
    target_stage: Optional[str] = None
    context: Optional[str] = None  # Additional context for retry


class SupervisorAgent:
    """
    Intelligent agent that interprets user feedback and decides actions.
    
    Handles both special commands (e.g., /goto, /retry) and natural language
    feedback using LLM-based intent classification.
    """
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def interpret_feedback(
        self, 
        feedback: str, 
        state: WorkflowState,
        available_stages: List[str]
    ) -> Action:
        """
        Interpret user feedback and return an action.
        
        Args:
            feedback: User's feedback text
            state: Current workflow state
            available_stages: List of valid stage names
        
        Returns:
            Action to take based on feedback
        """
        # Empty feedback = approval
        if not feedback or not feedback.strip():
            return Action(type="continue")
        
        feedback = feedback.strip()
        
        # Check for special commands
        if feedback.startswith("/"):
            return self.parse_command(feedback, available_stages)
        
        # Use LLM to interpret natural language feedback
        return self.classify_intent(feedback, state, available_stages)
    
    def parse_command(self, command: str, available_stages: List[str]) -> Action:
        """
        Parse special commands.
        
        Commands:
            /next - Continue to next stage
            /regenerate - Retry current stage
            /retry <msg> - Retry with additional context
            /goto <stage> - Jump to specific stage
            /quit - Exit workflow
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        
        if cmd == "/next":
            return Action(type="continue")
        
        elif cmd == "/regenerate":
            return Action(type="retry")
        
        elif cmd == "/retry":
            context = parts[1] if len(parts) > 1 else None
            return Action(type="retry", context=context)
        
        elif cmd == "/goto":
            if len(parts) < 2:
                # Invalid command, continue
                return Action(type="continue")
            
            target = parts[1].strip()
            # Validate stage exists
            if target in available_stages:
                return Action(type="goto", target_stage=target)
            else:
                # Invalid stage, continue
                return Action(type="continue")
        
        elif cmd == "/quit":
            return Action(type="quit")
        
        else:
            # Unknown command, treat as continue
            return Action(type="continue")
    
    def classify_intent(
        self, 
        feedback: str, 
        state: WorkflowState,
        available_stages: List[str]
    ) -> Action:
        """
        Use LLM to classify user's intent from natural language feedback.
        """
        # Get recent stage output for context
        last_record = state.stage_history[-1] if state.stage_history else None
        stage_context = ""
        if last_record:
            stage_context = f"Stage '{last_record.stage}' status: {last_record.status}"
        
        # Build classification prompt
        prompt = f"""You are a workflow supervisor analyzing user feedback.

Current stage: {state.current_stage}
{stage_context}

User feedback: "{feedback}"

Analyze the user's intent and classify it as one of the following:
1. "approve" - User is satisfied with the output and wants to continue to the next stage
2. "modify" - User wants specific changes or improvements to the current stage output
3. "regenerate" - User wants to completely redo the current stage
4. "navigate" - User wants to jump to a different stage in the workflow

Extract any specific context or suggestions the user mentioned.

Respond ONLY with a JSON object in this exact format:
{{
    "intent": "approve|modify|regenerate|navigate",
    "context": "specific feedback or changes requested",
    "reasoning": "brief explanation of classification"
}}

JSON response:"""
        
        try:
            # Call LLM
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages, tools=None)
            
            # Parse response
            content = response.content.strip()
            
            # Try to extract JSON from response
            if "```json" in content:
                # Extract from markdown code block
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                # Extract from generic code block
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            
            decision = json.loads(content)
            
            # Convert decision to action
            return self._decision_to_action(decision, feedback, available_stages)
        
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fallback: treat as modification request
            print(f"  [Supervisor] Could not parse LLM response, defaulting to retry: {e}")
            return Action(type="retry", context=feedback)
    
    def _decision_to_action(
        self, 
        decision: Dict, 
        original_feedback: str,
        available_stages: List[str]
    ) -> Action:
        """
        Convert LLM decision to Action.
        """
        intent = decision.get("intent", "modify")
        context = decision.get("context", original_feedback)
        
        if intent == "approve":
            return Action(type="continue")
        
        elif intent == "regenerate" or intent == "modify":
            return Action(type="retry", context=context)
        
        elif intent == "navigate":
            # Try to extract target stage from context
            # Simple heuristic: look for stage names in context
            target = None
            for stage_name in available_stages:
                if stage_name in context.lower():
                    target = stage_name
                    break
            
            if target:
                return Action(type="goto", target_stage=target)
            else:
                # No valid target found, just retry with context
                return Action(type="retry", context=context)
        
        else:
            # Default to retry
            return Action(type="retry", context=context)
