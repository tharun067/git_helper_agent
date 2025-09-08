# Git Repository AI Assistant

## Overview
The **Git Repository AI Assistant** is a Retrieval-Augmented Generation (RAG) pipeline that creates a conversational AI assistant for any public Git repository. It clones a repository, processes its contents, embeds them into a vector store using sentence transformers, and deploys a sophisticated agent built with **LangGraph** and **Groq’s Llama 3** to answer questions.

The agent is designed with a **routing mechanism** that directs user queries to specialized sub-agents (e.g., code-related, documentation, or general QA), ensuring accurate and context-aware responses.

---

## ✨ Key Features
- **Modular & Reusable**  
  Logical modules for ingestion, retrieval, and agent handling make it easy to extend and integrate.

- **Efficient Data Handling**  
  Clones a Git repository once, pulls updates on subsequent runs, and refreshes the vector store without rebuilding from scratch.

- **Intelligent Agent Routing**  
  Uses **LangGraph** to build a stateful multi-agent system that routes queries to the most appropriate sub-agent.

- **High-Performance LLM**  
  Powered by **Groq’s Llama 3 model** for real-time conversational responses.

- **Persistent Vector Storage**  
  Stores embeddings with **ChromaDB** locally, so re-embedding is only needed on repo updates.

- **Easy Configuration**  
  Centralized settings for repo URL, embedding model, and storage paths.

---
