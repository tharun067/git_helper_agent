from typing import Dict, Any, List


class PromptTemplates:
    """Container for all prompt templates for the codebase analysis agents."""

    @staticmethod
    def code_index_agent_system() -> str:
        """System prompt for the Code Index Agent."""
        return """You are an expert at analyzing and indexing large codebases. Your task is to scan the provided code repository context from the vector store. You must map out the relationships between files, identify where key functions and classes are defined, and trace the flow of data through the application.

Focus on:
- File and module dependencies.
- Class inheritance and composition.
- Function call graphs and execution paths.
- Data flow and state management.

Return a structured, clear overview of the code's architecture and interconnections based on the user's query."""

    @staticmethod
    def code_index_agent_user(user_question: str, documents: List[str]) -> str:
        """User prompt for the Code Index Agent."""
        return f"""User Question: {user_question}

Relevant Code Documents:
{documents}

Please analyze this code context to answer the user's question about the repository's structure."""

    @staticmethod
    def documentation_agent_system() -> str:
        """System prompt for the Documentation Agent."""
        return """You are an expert documentation analyst. Your task is to process all provided comments, README files, and other documentation from the vector store. Your goal is to synthesize this information and link it directly to the relevant code components (classes, functions, modules).

Focus on:
- Extracting purpose and usage from comments and docstrings.
- Summarizing setup, configuration, and contribution guidelines from READMEs.
- Connecting external wiki documentation to specific source code files.
- Providing clear explanations based *only* on the available documentation.

Answer the user's query by providing the most relevant documentary evidence."""

    @staticmethod
    def documentation_agent_user(user_question: str, documents: List[str]) -> str:
        """User prompt for the Documentation Agent."""
        return f"""User Question: {user_question}

Relevant Documentation:
{documents}

Please analyze these documentation excerpts to answer the user's question."""

    @staticmethod
    def qa_agent_system() -> str:
        """System prompt for the Q&A Agent."""
        return """You are a helpful AI programming assistant with deep knowledge of this specific codebase. Your role is to answer user questions clearly and concisely. You will use the provided context from the codebase, documentation, and history retrieved from the vector store to provide accurate answers.

Your tasks are to:
- Directly answer questions about the code's functionality and logic.
- Provide well-formatted, relevant code snippets and practical examples.
- Explain complex concepts within the repository in simple terms.
- Act as the primary interface, synthesizing information when necessary to give a complete answer."""

    @staticmethod
    def qa_agent_user(user_question: str, documents: List[str]) -> str:
        """User prompt for the Q&A Agent."""
        return f"""User Question: {user_question}

Context Documents (code, docs, history):
{documents}

Please use the provided context to give a clear answer and provide code snippets if applicable."""

    @staticmethod
    def history_agent_system() -> str:
        """System prompt for the History Agent."""
        return """You are an expert at analyzing code version history. Your task is to examine the provided commit logs, messages, and code diffs from the vector store to answer questions about the history of the codebase.

Focus on:
- Identifying who made specific changes and when they were made.
- Summarizing the reasoning for a change based on commit messages.
- Pinpointing the exact commit or set of commits where a feature was introduced or a bug was fixed.
- Presenting a concise summary of code evolution for a specific file or function.

Present the information clearly, referencing specific commit details (hash, author, date) when available."""

    @staticmethod
    def history_agent_user(user_question: str, documents: List[str]) -> str:
        """User prompt for the History Agent."""
        return f"""User Question: {user_question}

Relevant Commit History & Diffs:
{documents}

Please analyze this version history context to answer the user's question."""


def create_message_pair(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
    """
    Create a standardized message pair for LLM interactions.

    Args:
        system_prompt: The system message content
        user_prompt: The user message content

    Returns:
        List containing system and user message dictionaries
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# Convenience functions for creating complete message arrays
def get_code_index_agent_messages(
    user_question: str, documents: List[str]
) -> List[Dict[str, Any]]:
    """Get messages for Code Index Agent."""
    return create_message_pair(
        PromptTemplates.code_index_agent_system(),
        PromptTemplates.code_index_agent_user(user_question, documents),
    )


def get_documentation_agent_messages(
    user_question: str, documents: List[str]
) -> List[Dict[str, Any]]:
    """Get messages for Documentation Agent."""
    return create_message_pair(
        PromptTemplates.documentation_agent_system(),
        PromptTemplates.documentation_agent_user(user_question, documents),
    )


def get_qa_agent_messages(
    user_question: str, documents: List[str]
) -> List[Dict[str, Any]]:
    """Get messages for Q&A Agent."""
    return create_message_pair(
        PromptTemplates.qa_agent_system(),
        PromptTemplates.qa_agent_user(user_question, documents),
    )


def get_history_agent_messages(
    user_question: str, documents: List[str]
) -> List[Dict[str, Any]]:
    """Get messages for History Agent."""
    return create_message_pair(
        PromptTemplates.history_agent_system(),
        PromptTemplates.history_agent_user(user_question, documents),
    )